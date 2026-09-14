'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Message, { AssistantLoadingRow } from '@/components/Message';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

const HomePage = () => {
    const [input, setInput] = useState('');
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const scrollAnchorRef = useRef<HTMLDivElement>(null);
    const listRef = useRef<HTMLDivElement>(null);

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

    const { messages, sendMessage, status, error } = useChat({
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
        <main className="flex min-h-full flex-1 items-center justify-center bg-zinc-50 p-4 dark:bg-zinc-950">
            <div className="flex w-full max-w-2xl flex-col gap-4">
                <header className="px-1">
                    <h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
                        Research assistant
                    </h1>
                    <p className="text-sm text-zinc-500 dark:text-zinc-400">
                        Messages stream in real time. Reasoning appears in the
                        panel when the model provides it.
                    </p>
                </header>

                <div
                    ref={listRef}
                    className="flex h-[min(58vh,28rem)] flex-col gap-4 overflow-y-auto rounded-2xl border border-zinc-200/80 bg-white/90 p-4 shadow-sm backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/90"
                    aria-label="Chat messages"
                    aria-busy={isBusy}
                >
                    {messages.length === 0 ? (
                        <div className="flex h-full flex-col items-center justify-center text-center text-sm text-zinc-400">
                            <p>Ask a question to start the pipeline.</p>
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
                                label="Assistant is thinking…"
                            />
                        </div>
                    ) : null}
                    <div ref={scrollAnchorRef} className="h-px shrink-0" />
                </div>

                {error ? (
                    <p
                        className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300"
                        role="alert"
                    >
                        {error.message}
                    </p>
                ) : null}

                <form
                    onSubmit={handleSubmit}
                    className="flex flex-col gap-2 rounded-2xl border border-zinc-200/80 bg-white p-3 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
                >
                    <label htmlFor="chat-input" className="sr-only">
                        Message
                    </label>
                    <textarea
                        id="chat-input"
                        ref={inputRef}
                        rows={3}
                        autoFocus
                        placeholder="Ask anything… (Enter to send, Shift+Enter for newline)"
                        className="w-full resize-none rounded-xl border border-transparent bg-zinc-50 px-3 py-2 text-sm text-zinc-900 outline-none ring-zinc-300 transition focus:border-zinc-300 focus:ring-2 dark:bg-zinc-950 dark:text-zinc-100 dark:ring-zinc-700 dark:focus:border-zinc-600"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={isBusy}
                    />
                    <div className="flex items-center justify-between gap-2">
                        <span className="text-xs text-zinc-400">
                            {isBusy ? 'Streaming response…' : 'Ready'}
                        </span>
                        <button
                            type="submit"
                            disabled={isBusy || !input.trim()}
                            className="rounded-xl bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
                        >
                            Send
                        </button>
                    </div>
                </form>
            </div>
        </main>
    );
};

export default HomePage;
