import { toBaseMessages, toUIMessageStream } from '@ai-sdk/langchain';
import {
    createUIMessageStream,
    createUIMessageStreamResponse,
    type UIMessage,
} from 'ai';
import { HumanMessage } from '@langchain/core/messages';

import { workflow } from '@/graph/v1/pipeline';
import { createGraphStreamOptions } from '@/lib/graph-run-config';

export const maxDuration = 60;

export async function POST(request: Request) {
    let json: unknown;

    try {
        json = await request.json();
    } catch {
        return Response.json({ error: 'Invalid JSON body' }, { status: 400 });
    }

    const body = json as {
        messages?: UIMessage[];
        threadId?: string;
        id?: string;
    };

    const messages = body.messages;
    const threadId = body.threadId ?? body.id ?? 'default';

    if (!messages?.length) {
        return Response.json({ error: 'messages required' }, { status: 400 });
    }

    try {
        const langchainMessages = await toBaseMessages(messages);
        const incoming = langchainMessages.at(-1);

        if (!incoming || !HumanMessage.isInstance(incoming)) {
            return Response.json(
                { error: 'messages must end with a user turn' },
                { status: 400 },
            );
        }

        // UI Message Stream (SSE) — pairs with DefaultChatTransport on the client.
        // @see https://sdk.vercel.ai/docs/ai-sdk-ui/chatbot
        // @see https://sdk.vercel.ai/docs/reference/ai-sdk-ui/create-ui-message-stream
        return createUIMessageStreamResponse({
            stream: createUIMessageStream({
                originalMessages: messages,
                execute: async ({ writer }) => {
                    const graphStream = await workflow.stream(
                        { messages: [incoming] },
                        createGraphStreamOptions({ thread_id: threadId }),
                    );
                    writer.merge(toUIMessageStream(graphStream));
                },
            }),
        });
    } catch (error) {
        console.error('[api/v1/chat]', error);
        return Response.json(
            { error: 'Internal server error' },
            { status: 500 },
        );
    }
}
