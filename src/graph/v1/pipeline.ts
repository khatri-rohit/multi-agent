/**
 * Research pipeline (v1) — linear multi-agent flow with a single user-facing reply.
 *
 * Architecture (scalable pattern):
 * - `messages` (MessagesValue): **public** chat history only — human turns + final summary.
 * - `plan`, `researchNotes`: **private** deterministic handoffs between LLM stages.
 * - `researchMessages` (MessagesValue): **internal** agent↔tool loop (never streamed to UI).
 *
 * Flow:
 *   START → orchestrator → research ↔ researchTools → writer → END
 *
 * @see https://docs.langchain.com/oss/javascript/langgraph/use-graph-api#messagesvalue
 * @see https://docs.langchain.com/oss/javascript/langgraph/graph-api — separate state keys per concern
 */

import {
    StateGraph,
    StateSchema,
    MessagesValue,
    GraphNode,
    START,
    END,
    MemorySaver,
    InMemoryStore,
} from '@langchain/langgraph';
import {
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    isAIMessage,
} from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import z from 'zod';

import { getAgentModel } from '@/lib/model';
import { getSearchResults, scrapeWebsite } from '@/lib/ai/tools';

/** Hard cap on research LLM↔tool cycles to avoid runaway graphs in production. */
const MAX_RESEARCH_TOOL_ROUNDS = 8;

const State = new StateSchema({
    /** User-visible conversation (checkpointer + AI SDK UI). */
    messages: MessagesValue,
    /** Orchestrator output: what to research and how to frame the answer. */
    plan: z.string().optional(),
    /** Consolidated research text passed to the writer (set when research finishes). */
    researchNotes: z.string().optional(),
    /** Internal transcript for the research agent + ToolNode only. */
    researchMessages: MessagesValue,
    /** Guardrail counter for the research loop. */
    researchRounds: z.number().default(0),
});

const checkpointer = new MemorySaver();
const store = new InMemoryStore();

const model = getAgentModel();
const researchTools = [getSearchResults, scrapeWebsite];
const researchToolNode = new ToolNode(researchTools);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function lastHumanMessage(messages: BaseMessage[]): HumanMessage | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg instanceof HumanMessage) return msg;
    }
    return undefined;
}

function messageText(message: BaseMessage): string {
    const { content } = message;
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
        return content
            .map((part) => {
                if (typeof part === 'string') return part;
                if (part && typeof part === 'object' && 'text' in part) {
                    return String((part as { text: unknown }).text);
                }
                return '';
            })
            .join('\n')
            .trim();
    }
    return content != null ? String(content) : '';
}

/** Drop prior run's internal research transcript from checkpointed state. */
function clearResearchMessages(
    researchMessages: BaseMessage[],
): RemoveMessage[] {
    return researchMessages
        .filter((m) => m.id)
        .map((m) => new RemoveMessage({ id: m.id! }));
}

function pendingToolCalls(messages: BaseMessage[]) {
    const last = messages.at(-1);
    if (!last || !isAIMessage(last)) return [];
    return last.tool_calls ?? [];
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

/**
 * Orchestrator: reads the latest user turn, produces a plan for downstream agents.
 * Does **not** append to `messages` — avoids extra assistant bubbles in the UI.
 */
const orchestratorNode: GraphNode<typeof State> = async (state) => {
    const userTurn = lastHumanMessage(state.messages);
    if (!userTurn) {
        return {
            plan: 'No user message found.',
            researchNotes: '',
            researchRounds: 0,
            researchMessages: clearResearchMessages(state.researchMessages),
        };
    }

    const response = await model.invoke(
        [
            new SystemMessage(
                `You are the lead orchestrator for a research-and-summary pipeline.
Break down the user's request into a short, actionable plan for a research agent.
Include: goal, suggested search queries, and what the final summary should cover.
Do not answer the user directly — only output the plan.`,
            ),
            userTurn,
        ],
        { outputVersion: 'v1' },
    );

    return {
        plan: messageText(response),
        researchNotes: '',
        researchRounds: 0,
        researchMessages: clearResearchMessages(state.researchMessages),
    };
};

/**
 * Research agent: may call web tools. All model/tool traffic stays on `researchMessages`.
 */
const researchNode: GraphNode<typeof State> = async (state) => {
    const userTurn = lastHumanMessage(state.messages);
    const plan = state.plan ?? '';
    const rounds = state.researchRounds ?? 0;

    const seed = [
        new SystemMessage(
            `You are the research agent. Use tools when you need live web data.
Tools: get_search_urls (web search), scrape_websites (fetch page markdown).
Do not invent facts. When you have enough evidence, reply with a structured research brief (no more tool calls).

Orchestrator plan:
${plan}`,
        ),
        ...(userTurn ? [userTurn] : []),
        ...state.researchMessages,
    ];

    const response = await model
        .bindTools(researchTools)
        .invoke(seed, { outputVersion: 'v1' });

    const hasTools = (response.tool_calls?.length ?? 0) > 0;

    return {
        researchMessages: [response],
        researchRounds: rounds + 1,
        ...(hasTools ? {} : { researchNotes: messageText(response) }),
    };
};

/**
 * Runs tools requested by the last message on the **internal** research channel.
 * ToolNode expects a \`messages\` key; we bridge from \`researchMessages\`.
 */
const researchToolsNode: GraphNode<typeof State> = async (state) => {
    const output = await researchToolNode.invoke({
        messages: state.researchMessages,
    });
    return { researchMessages: output.messages };
};

/**
 * Writer: sole node that appends to public \`messages\` — one assistant reply per user turn.
 */
const writerNode: GraphNode<typeof State> = async (state) => {
    const userTurn = lastHumanMessage(state.messages);
    const plan = state.plan ?? '';
    const notes = state.researchNotes ?? 'No research notes were produced.';

    const response = await model.invoke(
        [
            new SystemMessage(
                `You are the technical writer. Produce the final user-facing answer as a clear, concise summary.
Use only the orchestrator plan and research notes below. If research is thin, say what is missing.
This is the only message the user will see for this turn.`,
            ),
            ...(userTurn ? [userTurn] : []),
            new HumanMessage(
                `Plan:\n${plan}\n\nResearch notes:\n${notes}`,
            ),
        ],
        { outputVersion: 'v1' },
    );

    return { messages: [response] };
};

// ---------------------------------------------------------------------------
// Routing (conditional edges follow LangGraph prebuilt agent pattern)
// @see https://docs.langchain.com/langsmith/setup-javascript#define-graphs
// ---------------------------------------------------------------------------

/**
 * Exact destinations allowed after `research`.
 * Conditional routers must return one of these strings — LangGraph does not
 * rename/alias nodes. Mismatches throw:
 * "Branch condition returned unknown or null destination".
 *
 * @see https://docs.langchain.com/oss/javascript/langgraph/graph-api#conditional-edges
 */
const AFTER_RESEARCH = ['researchTools', 'writer'] as const;
type AfterResearch = (typeof AFTER_RESEARCH)[number];

function routeAfterResearch(state: typeof State.State): AfterResearch {
    if ((state.researchRounds ?? 0) >= MAX_RESEARCH_TOOL_ROUNDS) {
        return 'writer';
    }
    if (pendingToolCalls(state.researchMessages).length > 0) {
        return 'researchTools';
    }
    return 'writer';
}

function routeAfterResearchTools(): 'research' {
    return 'research';
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export const workflow = new StateGraph(State)
    .addNode('orchestrator', orchestratorNode)
    .addNode('research', researchNode)
    .addNode('researchTools', researchToolsNode)
    .addNode('writer', writerNode)
    .addEdge(START, 'orchestrator')
    .addEdge('orchestrator', 'research')
    .addConditionalEdges('research', routeAfterResearch, [...AFTER_RESEARCH])
    .addConditionalEdges('researchTools', routeAfterResearchTools, ['research'])
    .addEdge('writer', END)
    .compile({ checkpointer, store });

export const graph = workflow;

export type ResearchPipelineGraph = typeof graph;
