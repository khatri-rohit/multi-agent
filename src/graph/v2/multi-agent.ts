/**
 * Multi-agent research team (v2) — hub-and-spoke supervisor + research ReAct loop.
 *
 * Flow:
 *   START → supervisor ⇄ { planning | research ⇄ tools | analysis | writer } → END
 *
 * Supervisor is the only node that may go to END. Research uses conditional edges
 * for the tool loop; all other specialists return via Command to the supervisor.
 *
 * @see https://docs.langchain.com/oss/javascript/langgraph/use-graph-api#command
 * @see https://docs.langchain.com/oss/javascript/langgraph/workflows-agents
 */

import {
    StateGraph,
    StateSchema,
    MessagesValue,
    GraphNode,
    Command,
    START,
    END,
    MemorySaver,
} from '@langchain/langgraph';
import {
    AIMessage,
    HumanMessage,
    SystemMessage,
    isAIMessage,
} from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import z from 'zod';

import { getAgentModel } from '@/lib/model';
import { getSearchResults, scrapeWebsite } from '@/lib/ai/tools';

const SupervisorRoute = z.object({
    next: z.enum(['planning', 'research', 'analysis', 'writer', 'end']),
    instruction: z.string(),
});

type SupervisorNext = z.infer<typeof SupervisorRoute>['next'];

const State = new StateSchema({
    messages: MessagesValue,
    planningData: z.string().default(''),
    researchData: z.string().default(''),
    analysisData: z.string().default(''),
    finalReport: z.string().default(''),
    currentTask: z.string().default(''),
});

const checkpointer = new MemorySaver();
const model = getAgentModel();

const researchTools = [getSearchResults, scrapeWebsite];
const researchToolsNode = new ToolNode(researchTools, {
    handleToolErrors: true,
});

function artifactPreview(value: string | undefined, emptyLabel: string): string {
    const trimmed = value?.trim() ?? '';
    return trimmed.length > 0 ? trimmed : emptyLabel;
}

function lastHumanText(messages: typeof State.State['messages']): string {
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (HumanMessage.isInstance(msg)) {
            return typeof msg.content === 'string'
                ? msg.content
                : JSON.stringify(msg.content);
        }
    }
    return '';
}

function messageContentText(content: unknown): string {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
        return content
            .map((part) => {
                if (typeof part === 'string') return part;
                if (
                    part &&
                    typeof part === 'object' &&
                    'text' in part &&
                    typeof (part as { text: unknown }).text === 'string'
                ) {
                    return (part as { text: string }).text;
                }
                return '';
            })
            .join('');
    }
    return content == null ? '' : String(content);
}

const SUPERVISOR_NEXT = [
    'planning',
    'research',
    'analysis',
    'writer',
    'end',
] as const;

/**
 * Recover a route when the model ignores JSON format and returns prose.
 * Handles both `{"next":...}` blobs and phrases like "Next specialist: planning".
 */
function parseSupervisorDecision(
    text: string,
    fallback: z.infer<typeof SupervisorRoute>,
): z.infer<typeof SupervisorRoute> {
    const trimmed = text.trim();
    if (!trimmed) return fallback;

    const candidates: unknown[] = [];
    try {
        candidates.push(JSON.parse(trimmed));
    } catch {
        // continue
    }

    const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced?.[1]) {
        try {
            candidates.push(JSON.parse(fenced[1].trim()));
        } catch {
            // continue
        }
    }

    const brace = trimmed.match(/\{[\s\S]*\}/);
    if (brace?.[0]) {
        try {
            candidates.push(JSON.parse(brace[0]));
        } catch {
            // continue
        }
    }

    for (const candidate of candidates) {
        if (!candidate || typeof candidate !== 'object') continue;
        const record = candidate as Record<string, unknown>;
        const nextRaw = record.next ?? record.nextAgent ?? record.agent;
        const instructionRaw =
            record.instruction ?? record.task ?? record.currentTask;
        const parsed = SupervisorRoute.safeParse({
            next:
                typeof nextRaw === 'string'
                    ? nextRaw.toLowerCase().trim()
                    : nextRaw,
            instruction:
                typeof instructionRaw === 'string'
                    ? instructionRaw.trim()
                    : '',
        });
        if (parsed.success) return parsed.data;
    }

    const nextMatch =
        trimmed.match(
            /(?:next(?:\s+specialist)?|route(?:\s+to)?)\s*[:\-]?\s*\*{0,2}(planning|research|analysis|writer|end)\b/i,
        ) ??
        trimmed.match(
            /\*{0,2}(planning|research|analysis|writer|end)\*{0,2}\s+specialist/i,
        ) ??
        trimmed.match(
            /\b(planning|research|analysis|writer|end)\b/i,
        );

    const instructionMatch = trimmed.match(
        /\*{0,2}instruction\*{0,2}\s*:\*{0,2}\s*([\s\S]+)/i,
    );

    const next = (nextMatch?.[1]?.toLowerCase() ??
        fallback.next) as SupervisorNext;
    const instruction =
        instructionMatch?.[1]
            ?.trim()
            .replace(/^\*{1,2}\s*/, '')
            .replace(/\n{2,}[\s\S]*$/, '')
            .trim() || fallback.instruction;

    if ((SUPERVISOR_NEXT as readonly string[]).includes(next)) {
        return { next, instruction };
    }

    return fallback;
}

/** Artifact-driven default when the model output is unusable. */
function defaultSupervisorDecision(
    state: typeof State.State,
): z.infer<typeof SupervisorRoute> {
    const userTask =
        lastHumanText(state.messages) ||
        state.currentTask ||
        'Complete the assigned work.';

    if (!state.planningData?.trim()) {
        return {
            next: 'planning',
            instruction: `Create a concrete research plan for: ${userTask}`,
        };
    }
    if (!state.researchData?.trim()) {
        return {
            next: 'research',
            instruction: `Execute the research plan and gather grounded facts for: ${userTask}`,
        };
    }
    if (!state.analysisData?.trim()) {
        return {
            next: 'analysis',
            instruction:
                'Analyze the research for completeness and accuracy. Flag gaps.',
        };
    }
    if (!state.finalReport?.trim()) {
        return {
            next: 'writer',
            instruction:
                'Write the final report using only the analysis. Do not invent facts.',
        };
    }
    return { next: 'end', instruction: 'Task complete.' };
}

/**
 * Ask the model for a route. Prefer Ollama JSON format; fall back to prose parse.
 * Never throw OUTPUT_PARSING_FAILURE up to the graph.
 */
async function invokeSupervisorDecision(
    state: typeof State.State,
): Promise<z.infer<typeof SupervisorRoute>> {
    const fallback = defaultSupervisorDecision(state);
    const userTask = lastHumanText(state.messages) || state.currentTask || 'No task';

    const messages = [
        new SystemMessage(`You are the supervisor of a research team.
Specialists:
- planning: produce a concrete research plan
- research: search and scrape facts using tools (no invented facts)
- analysis: check research completeness and accuracy (no new facts)
- writer: write the final report from analysis
- end: only when a final report already exists

Current artifacts:
- planningData: ${artifactPreview(state.planningData, '(empty)')}
- researchData: ${artifactPreview(state.researchData, '(empty)')}
- analysisData: ${artifactPreview(state.analysisData, '(empty)')}
- finalReport: ${artifactPreview(state.finalReport, '(empty)')}

Prefer the natural order: planning → research → analysis → writer → end.
You may send research again after analysis if evidence is thin.

Respond with ONLY a JSON object. No markdown. No prose. No code fences.
Example: {"next":"planning","instruction":"Create a research plan for ..."}`),
        new HumanMessage(userTask),
    ];

    // glm / some Ollama cloud models ignore JSON format and return prose.
    // Never use withStructuredOutput here — it throws OUTPUT_PARSING_FAILURE
    // on prose. Parse ourselves (JSON blob or "Next specialist: …" text).
    try {
        const raw = await model
            .withConfig({ format: 'json' })
            .invoke(messages);
        return parseSupervisorDecision(
            messageContentText(raw.content),
            fallback,
        );
    } catch (error) {
        console.warn(
            'Supervisor JSON invoke failed; retrying without format constraint.',
            error,
        );
    }

    try {
        const raw = await model.invoke(messages);
        return parseSupervisorDecision(
            messageContentText(raw.content),
            fallback,
        );
    } catch (error) {
        console.warn(
            'Supervisor invoke failed; using artifact fallback.',
            error,
        );
        return fallback;
    }
}

/**
 * Deterministic policy: never skip a missing upstream artifact.
 * Model may still choose to re-run research after analysis exists.
 */
function resolveSupervisorRoute(
    state: typeof State.State,
    decision: z.infer<typeof SupervisorRoute>,
): { next: SupervisorNext | typeof END; instruction: string } {
    const hasPlan = !!state.planningData?.trim();
    const hasResearch = !!state.researchData?.trim();
    const hasAnalysis = !!state.analysisData?.trim();
    const hasReport = !!state.finalReport?.trim();

    if (hasReport) {
        return { next: END, instruction: 'Task complete.' };
    }

    let next: SupervisorNext = decision.next;
    let instruction = decision.instruction.trim() || state.currentTask;

    if (!hasPlan) {
        next = 'planning';
    } else if (!hasResearch) {
        next = 'research';
    } else if (!hasAnalysis) {
        // Allow supervisor to request more research even after a plan exists.
        next = decision.next === 'research' ? 'research' : 'analysis';
    } else if (!hasReport) {
        // Allow another research hop after analysis if the supervisor asks.
        if (decision.next === 'research') {
            next = 'research';
        } else if (decision.next === 'analysis') {
            next = 'analysis';
        } else if (decision.next === 'end') {
            // No report yet — cannot end.
            next = 'writer';
        } else {
            next = 'writer';
        }
    }

    // Block skipping upstream artifacts even if the model asks.
    if (next === 'writer' && !hasAnalysis) next = hasResearch ? 'analysis' : 'research';
    if (next === 'analysis' && !hasResearch) next = 'research';
    if (next === 'research' && !hasPlan) next = 'planning';
    if (next === 'end' && !hasReport) {
        next = !hasPlan
            ? 'planning'
            : !hasResearch
              ? 'research'
              : !hasAnalysis
                ? 'analysis'
                : 'writer';
    }

    if (!instruction) {
        instruction = lastHumanText(state.messages) || 'Complete the assigned work.';
    }

    return { next: next === 'end' ? END : next, instruction };
}

const supervisorAgent: GraphNode<typeof State> = async (state) => {
    if (state.finalReport?.trim()) {
        return new Command({
            goto: END,
            update: {
                messages: [
                    new AIMessage({
                        name: 'supervisor',
                        content: 'Supervisor: Final report ready. Ending.',
                    }),
                ],
            },
        });
    }

    const decision = await invokeSupervisorDecision(state);
    const { next, instruction } = resolveSupervisorRoute(state, decision);
    const label = next === END ? 'end' : next;

    return new Command({
        goto: next,
        update: {
            currentTask: instruction,
            messages: [
                new AIMessage({
                    name: 'supervisor',
                    content: `Supervisor: routing to ${label}. Instruction: ${instruction}`,
                }),
            ],
        },
    });
};

const planningAgent: GraphNode<typeof State> = async (state) => {
    const task = state.currentTask || lastHumanText(state.messages);

    const response = await model.invoke([
        new SystemMessage(`You are a specialized planning agent.
Create a detailed, specific research plan for the task.
The research agent can use get_search_urls and scrape_websites.
Do not invent facts — outline what to look up and how.`),
        new HumanMessage(`Task: ${task}`),
    ]);

    const content = messageContentText(response.content);
    console.log('Planning:', content);

    return new Command({
        goto: 'supervisor',
        update: {
            planningData: content,
            messages: [
                new AIMessage({
                    name: 'planning',
                    content,
                }),
            ],
        },
    });
};

const researchAgent: GraphNode<typeof State> = async (state) => {
    const task = state.currentTask || lastHumanText(state.messages);
    const planningData = state.planningData;

    const response = await model.bindTools(researchTools).invoke([
        new SystemMessage(`You are a research agent.
Use tools get_search_urls and scrape_websites. Do not invent facts.
Base your work on this plan:
${artifactPreview(planningData, '(no plan yet)')}

Current instruction: ${task}

When you have enough grounded evidence, respond with a concise research summary covering:
1. Key facts and background
2. Current trends and developments
3. Important statistics or data
4. Notable examples or case studies (if any)

If you still need information, call tools.`),
        ...state.messages,
    ]);

    console.log('Research:', response.content);

    const update: Partial<typeof State.State> = {
        messages: [response],
    };

    // Only persist the grounded summary when the model is done calling tools.
    if (!response.tool_calls?.length) {
        update.researchData = messageContentText(response.content);
    }

    return update;
};

const analysisAgent: GraphNode<typeof State> = async (state) => {
    const response = await model.invoke([
        new SystemMessage(`You are an analysis agent.
Analyze the research data for completeness and accuracy.
Do not invent new information. Flag gaps if evidence is thin.`),
        new HumanMessage(
            `Research data:\n${artifactPreview(state.researchData, '(empty)')}`,
        ),
    ]);

    const content = messageContentText(response.content);
    console.log('Analysis:', content);

    return new Command({
        goto: 'supervisor',
        update: {
            analysisData: content,
            messages: [
                new AIMessage({
                    name: 'analysis',
                    content,
                }),
            ],
        },
    });
};

const writerAgent: GraphNode<typeof State> = async (state) => {
    const response = await model.invoke([
        new SystemMessage(`You are a writer agent.
Write a clear, well-structured final report based only on the analysis.
Do not invent facts.`),
        new HumanMessage(
            `Analysis:\n${artifactPreview(state.analysisData, '(empty)')}`,
        ),
    ]);

    const content = messageContentText(response.content);
    console.log('Writer:', content);

    return new Command({
        goto: 'supervisor',
        update: {
            finalReport: content,
            messages: [
                new AIMessage({
                    name: 'writer',
                    content,
                }),
            ],
        },
    });
};

const researchRouter = (state: typeof State.State): 'tools' | 'supervisor' => {
    const last = state.messages.at(-1);
    if (last && isAIMessage(last) && last.tool_calls?.length) {
        return 'tools';
    }
    return 'supervisor';
};

export const workflow = new StateGraph(State)
    .addNode('supervisor', supervisorAgent, {
        ends: ['planning', 'research', 'analysis', 'writer', END],
    })
    .addNode('planning', planningAgent, { ends: ['supervisor'] })
    .addNode('research', researchAgent)
    .addNode('tools', researchToolsNode)
    .addNode('analysis', analysisAgent, { ends: ['supervisor'] })
    .addNode('writer', writerAgent, { ends: ['supervisor'] })
    .addEdge(START, 'supervisor')
    .addConditionalEdges('research', researchRouter, ['tools', 'supervisor'])
    .addEdge('tools', 'research')
    .compile({ checkpointer });
