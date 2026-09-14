import type { UIMessage } from 'ai';

type MessageProps = {
    message: UIMessage;
    /** True while this message is the active assistant turn being streamed */
    isStreaming?: boolean;
};

function collectReasoning(parts: UIMessage['parts']) {
    const chunks = parts.filter((p) => p.type === 'reasoning');
    const text = chunks.map((p) => p.text).join('\n\n').trim();
    const isReasoningStreaming = chunks.some((p) => p.state === 'streaming');
    return { text, hasReasoning: chunks.length > 0, isReasoningStreaming };
}

function collectResponseText(parts: UIMessage['parts']) {
    return parts
        .filter((p) => p.type === 'text')
        .map((p) => p.text)
        .join('')
        .trim();
}

function ToolPart({
    part,
    index,
    messageId,
}: {
    part: UIMessage['parts'][number];
    index: number;
    messageId: string;
}) {
    if (
        part.type !== 'dynamic-tool' &&
        !part.type.startsWith('tool-')
    ) {
        return null;
    }

    const toolPart = part as {
        type: string;
        state?: string;
        input?: unknown;
    };
    const name =
        part.type === 'dynamic-tool'
            ? 'tool'
            : part.type.replace(/^tool-/, '');
    const running =
        toolPart.state !== 'output-available' &&
        toolPart.state !== 'output-error';

    return (
        <div
            key={`${messageId}-tool-${index}`}
            className="rounded-lg border border-zinc-200/80 bg-white/80 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/80 dark:text-zinc-400"
        >
            <span className="font-medium text-zinc-800 dark:text-zinc-200">
                {name}
            </span>
            <span
                className={`ml-2 rounded-full px-2 py-0.5 text-[10px] uppercase ${
                    running
                        ? 'bg-sky-500/15 text-sky-700 dark:text-sky-300'
                        : 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
                }`}
            >
                {running ? 'running' : 'done'}
            </span>
        </div>
    );
}

export function AssistantLoadingRow({ label }: { label: string }) {
    return (
        <div
            className="flex items-center gap-2 py-2 text-sm text-zinc-500 dark:text-zinc-400"
            role="status"
            aria-live="polite"
        >
            <span className="inline-flex gap-1">
                <span className="size-1.5 animate-bounce rounded-full bg-zinc-400 [animation-delay:0ms]" />
                <span className="size-1.5 animate-bounce rounded-full bg-zinc-400 [animation-delay:150ms]" />
                <span className="size-1.5 animate-bounce rounded-full bg-zinc-400 [animation-delay:300ms]" />
            </span>
            <span>{label}</span>
        </div>
    );
}

const Message = ({ message, isStreaming = false }: MessageProps) => {
    if (message.role === 'user') {
        const text = collectResponseText(message.parts);
        if (!text) return null;

        return (
            <div className="flex justify-end">
                <div
                    className="max-w-[85%] rounded-2xl rounded-br-md bg-zinc-900 px-4 py-2.5 text-sm leading-relaxed text-white dark:bg-zinc-100 dark:text-zinc-900"
                >
                    <p className="whitespace-pre-wrap">{text}</p>
                </div>
            </div>
        );
    }

    const { text: reasoningText, hasReasoning, isReasoningStreaming } =
        collectReasoning(message.parts);
    const responseText = collectResponseText(message.parts);
    const hasToolParts = message.parts.some(
        (p) => p.type === 'dynamic-tool' || p.type.startsWith('tool-'),
    );

    const waitingForResponse =
        isStreaming && !responseText && !hasToolParts;
    const loadingLabel =
        hasReasoning && (isReasoningStreaming || waitingForResponse)
            ? 'Composing response…'
            : 'Assistant is thinking…';

    const lastTextPartIndex = message.parts.reduce(
        (last, part, index) => (part.type === 'text' ? index : last),
        -1,
    );

    return (
        <div className="flex justify-start">
            <div className="flex w-full max-w-[92%] flex-col gap-2">
                {hasReasoning && (
                    <div className="overflow-hidden rounded-xl border border-violet-200/80 bg-violet-50/90 dark:border-violet-900/60 dark:bg-violet-950/40">
                        <div className="border-b border-violet-200/60 px-3 py-1.5 text-[11px] font-medium uppercase tracking-wide text-violet-700 dark:border-violet-800/60 dark:text-violet-300">
                            Reasoning
                            {isReasoningStreaming && isStreaming ? (
                                <span className="ml-2 font-normal normal-case text-violet-500">
                                    streaming
                                </span>
                            ) : null}
                        </div>
                        <div
                            className="h-28 overflow-y-auto px-3 py-2 font-mono text-xs leading-relaxed text-violet-950/90 dark:text-violet-100/90"
                        >
                            <p className="whitespace-pre-wrap">
                                {reasoningText}
                                {isReasoningStreaming && isStreaming ? (
                                    <span
                                        className="ml-0.5 inline-block h-3 w-0.5 animate-pulse bg-violet-500 align-middle"
                                        aria-hidden
                                    />
                                ) : null}
                            </p>
                        </div>
                    </div>
                )}

                {isStreaming && (waitingForResponse || isReasoningStreaming) ? (
                    <AssistantLoadingRow label={loadingLabel} />
                ) : null}

                {message.parts.map((part, index) => {
                    if (part.type === 'text' && part.text) {
                        return (
                            <div
                                key={`${message.id}-text-${index}`}
                                className="rounded-2xl rounded-bl-md border border-zinc-200/80 bg-white px-4 py-3 text-sm leading-relaxed text-zinc-800 shadow-sm dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
                            >
                                <p className="whitespace-pre-wrap">
                                    {part.text}
                                    {isStreaming &&
                                    index === lastTextPartIndex ? (
                                        <span
                                            className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-zinc-400 align-middle"
                                            aria-hidden
                                        />
                                    ) : null}
                                </p>
                            </div>
                        );
                    }

                    return (
                        <ToolPart
                            key={`${message.id}-tool-wrap-${index}`}
                            part={part}
                            index={index}
                            messageId={message.id}
                        />
                    );
                })}

                {!responseText &&
                    !hasReasoning &&
                    !hasToolParts &&
                    isStreaming && (
                        <AssistantLoadingRow label="Assistant is thinking…" />
                    )}
            </div>
        </div>
    );
};

export default Message;
