'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Message, { AssistantLoadingRow } from '@/components/Message';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport, type ChatStatus } from 'ai';

/**
 * Maps AI SDK chat status to a short UI label while the pipeline runs.
 * @see https://sdk.vercel.ai/docs/ai-sdk-ui/chatbot#status
 */
function streamStatusLabel(status: ChatStatus): string {
    switch (status) {
        case 'submitted':
            return 'Pipeline running (orchestrator → research → writer)…';
        case 'streaming':
            return 'Streaming response';
        case 'error':
            return 'Error';
        default:
            return 'Ready';
    }
}

const HomePage = () => {
    const [input, setInput] = useState('');
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const scrollAnchorRef = useRef<HTMLDivElement>(null);
    const listRef = useRef<HTMLDivElement>(null);

    /**
     * DefaultChatTransport consumes the UI Message Stream (SSE) from
     * `createUIMessageStreamResponse` — do not use TextStreamChatTransport here.
     * @see https://sdk.vercel.ai/docs/ai-sdk-ui/transport
     */
    const transport = useMemo(
        () =>
            new DefaultChatTransport({
                api: '/api/v1/chat',
                prepareSendMessagesRequest: ({ id, messages }) => ({
                    body: {
                        threadId: id,
                        messages,
                    },
                }),
            }),
        [],
    );

    const { messages, sendMessage, status, error, stop } = useChat({
        id: 'pipeline-chat',
        transport,
    });

    const isBusy = status === 'streaming' || status === 'submitted';

    useEffect(() => {
        scrollAnchorRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, status]);

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const text = input.trim();
        if (!text || isBusy) return;
        void sendMessage({ text });
        setInput('');
        inputRef.current?.focus();
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            e.currentTarget.form?.requestSubmit();
        }
    };

    const lastMessage = messages.at(-1);
    const awaitingAssistantRow =
        status === 'streaming' && lastMessage?.role === 'user';

    return (
        <div className="flex h-dvh min-h-0 flex-col bg-zinc-50 dark:bg-zinc-950">
            <header
                className="shrink-0 border-b border-zinc-200/80 bg-white/90 px-4 py-3 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/90"
            >
                <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4">
                    <div className="min-w-0">
                        <h1 className="truncate text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
                            Research assistant
                        </h1>
                        <p className="truncate text-xs text-zinc-500 dark:text-zinc-400">
                            Full-page chat · UI message stream (SSE)
                        </p>
                    </div>
                    <div
                        className="flex shrink-0 items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400"
                        aria-live="polite"
                    >
                        <span
                            className={`size-2 rounded-full ${
                                status === 'streaming'
                                    ? 'animate-pulse bg-emerald-500'
                                    : status === 'submitted'
                                      ? 'animate-pulse bg-amber-500'
                                      : status === 'error'
                                        ? 'bg-red-500'
                                        : 'bg-zinc-300 dark:bg-zinc-600'
                            }`}
                            aria-hidden
                        />
                        <span className="max-w-[14rem] truncate sm:max-w-none">
                            {streamStatusLabel(status)}
                        </span>
                    </div>
                </div>
            </header>

            <div
                ref={listRef}
                className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-6"
                aria-label="Chat messages"
                aria-busy={isBusy}
            >
                <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
                    {messages.length === 0 ? (
                        <div className="flex min-h-[40vh] flex-col items-center justify-center text-center">
                            <p className="text-sm font-medium text-zinc-600 dark:text-zinc-300">
                                Ask a question to start the pipeline
                            </p>
                            <p className="mt-2 max-w-md text-xs leading-relaxed text-zinc-400">
                                One assistant reply per turn. Orchestrator and
                                research run in the background; the writer
                                streams the final summary here.
                            </p>
                        </div>
                    ) : (
                        messages.map((message) => (
                            <Message
                                key={message.id}
                                message={message}
                                isStreaming={
                                    status === 'streaming' &&
                                    message.role === 'assistant' &&
                                    message.id === lastMessage?.id
                                }
                            />
                        ))
                    )}
                    {awaitingAssistantRow ? (
                        <div className="flex justify-start">
                            <AssistantLoadingRow
                                label="Running research pipeline…"
                            />
                        </div>
                    ) : null}
                    <div ref={scrollAnchorRef} className="h-px shrink-0" />
                </div>
            </div>

            <footer
                className="shrink-0 border-t border-zinc-200/80 bg-white/95 px-4 py-4 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/95"
            >
                <div className="mx-auto w-full max-w-3xl">
                    {error ? (
                        <p
                            className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300"
                            role="alert"
                        >
                            {error.message}
                        </p>
                    ) : null}

                    <form
                        onSubmit={handleSubmit}
                        className="flex flex-col gap-2 rounded-2xl border border-zinc-200/80 bg-zinc-50/80 p-3 dark:border-zinc-700 dark:bg-zinc-950/80"
                    >
                        <label htmlFor="chat-input" className="sr-only">
                            Message
                        </label>
                        <textarea
                            id="chat-input"
                            ref={inputRef}
                            rows={2}
                            autoFocus
                            placeholder="Ask anything… (Enter to send, Shift+Enter for newline)"
                            className="w-full resize-none rounded-xl border border-transparent bg-white px-3 py-2.5 text-sm text-zinc-900 outline-none ring-zinc-300 transition focus:border-zinc-300 focus:ring-2 dark:bg-zinc-900 dark:text-zinc-100 dark:ring-zinc-700 dark:focus:border-zinc-600"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={isBusy}
                        />
                        <div className="flex items-center justify-between gap-2">
                            <span className="text-xs text-zinc-400">
                                {streamStatusLabel(status)}
                            </span>
                            <div className="flex items-center gap-2">
                                {status === 'streaming' ? (
                                    <button
                                        type="button"
                                        onClick={() => stop()}
                                        className="rounded-xl border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800"
                                    >
                                        Stop
                                    </button>
                                ) : null}
                                <button
                                    type="submit"
                                    disabled={isBusy || !input.trim()}
                                    className="rounded-xl bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
                                >
                                    Send
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </footer>
        </div>
    );
};

export default HomePage;
