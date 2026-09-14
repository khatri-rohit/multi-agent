import { toBaseMessages, toUIMessageStream } from '@ai-sdk/langchain';
import { createUIMessageStreamResponse, type UIMessage } from 'ai';
import { HumanMessage } from '@langchain/core/messages';

import { graph } from '@/graph/v1/pipeline';

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

        const stream = await graph.stream(
            { messages: [incoming] },
            {
                streamMode: ['values', 'messages', 'updates'],
                configurable: { thread_id: threadId },
            },
        );

        return createUIMessageStreamResponse({
            stream: toUIMessageStream(stream),
        });
    } catch (error) {
        console.error('[api/v1/chat]', error);
        return Response.json(
            { error: 'Internal server error' },
            { status: 500 },
        );
    }
}
