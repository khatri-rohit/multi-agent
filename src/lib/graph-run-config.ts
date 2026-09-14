/**
 * Shared LangGraph streaming options for Next.js chat routes.
 *
 * `toUIMessageStream` from `@ai-sdk/langchain` expects these stream modes so the
 * Vercel AI SDK can map graph output to a single assistant UI message.
 *
 * @see https://docs.langchain.com/oss/javascript/langgraph/streaming
 * @see https://docs.langchain.com/oss/javascript/langgraph/add-memory — stable `thread_id`
 */
export const GRAPH_CHAT_STREAM_MODES = [
    'values',
    'messages',
    'updates',
] as const;

export type GraphChatStreamMode = (typeof GRAPH_CHAT_STREAM_MODES)[number];

export type GraphRunConfigurable = {
    thread_id: string;
};

export function createGraphStreamOptions(configurable: GraphRunConfigurable) {
    return {
        streamMode: [...GRAPH_CHAT_STREAM_MODES] as GraphChatStreamMode[],
        recursionLimit: 25,
        configurable: {
            thread_id: configurable.thread_id,
        },
    };
}
