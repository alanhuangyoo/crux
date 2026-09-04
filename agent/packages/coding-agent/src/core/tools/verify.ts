import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type Static, Type } from "typebox";
import type { Theme } from "../../modes/interactive/theme/theme.ts";
import type { ExtensionContext, ToolDefinition, ToolRenderResultOptions } from "../extensions/types.ts";
import { getTextOutput } from "./render-utils.ts";

/**
 * verify -- bind a claim about the work to a command that proves it.
 *
 * The measurement this exists for: on Terminal-Bench 2.1 with a self-hosted
 * Qwen3.8-27B, 19 of this agent's 24 failures were trials that ran to
 * completion and answered wrongly -- not timeouts, not crashes. The agent
 * believed it was done. More prompting does not fix that, because from inside
 * the turn every individual check already looks sufficient.
 *
 * What changes the outcome is a mechanism rather than an instruction: a claim
 * is only closed by a command that re-runs and passes, and finishing re-runs
 * all of them at once. That is what catches the requirement met at step 10 and
 * quietly broken at step 40 -- the single most common way a task fails while
 * looking finished.
 *
 * The checklist is per-session and in-memory on purpose: it describes the work
 * in front of the agent, not a durable artifact, and a stale checklist that
 * outlived its task would pass on nothing.
 */

const run = promisify(execFile);

// Long enough for a build or a test suite, short enough that one hung check
// does not eat the turn. A verification that needs longer is doing something
// other than verifying.
const CHECK_TIMEOUT_MS = 120_000;
const MAX_DETAIL = 400;

const verifySchema = Type.Object({
	action: Type.Union([Type.Literal("check"), Type.Literal("submit"), Type.Literal("list")], {
		description:
			"check: bind a claim to a command and run it now. submit: re-run every bound check. list: show them.",
	}),
	claim: Type.Optional(Type.String({ description: "What must be true, in one line (action=check)" })),
	command: Type.Optional(
		Type.String({ description: "Shell command that exits 0 only if the claim holds (action=check)" }),
	),
});

export type VerifyToolInput = Static<typeof verifySchema>;

interface Item {
	claim: string;
	command: string;
	passing: boolean;
	detail: string;
}

export interface VerifyToolDetails {
	items: Item[];
	failing: number;
}

export const verifyToolSystemPromptContribution = {
	snippet: "Bind claims about the work to commands that prove them, and re-run them all before finishing",
	guidelines: [],
} as const;

export interface VerifyToolOptions {
	/** Override how a check is executed. Default: bash -lc in the session cwd. */
	runner?: (command: string, cwd: string) => Promise<{ passing: boolean; detail: string }>;
}

async function defaultRunner(command: string, cwd: string): Promise<{ passing: boolean; detail: string }> {
	try {
		const { stdout, stderr } = await run("bash", ["-lc", command], {
			cwd,
			timeout: CHECK_TIMEOUT_MS,
			maxBuffer: 1024 * 1024,
		});
		return { passing: true, detail: (stdout + stderr).trim().slice(-MAX_DETAIL) };
	} catch (err: any) {
		// A non-zero exit is the signal, not an accident: the command is written
		// so that failure means the claim does not hold.
		const detail = [err?.stdout, err?.stderr, err?.message].filter(Boolean).join("\n").trim().slice(-MAX_DETAIL);
		const timedOut = err?.killed === true || err?.signal === "SIGTERM";
		return { passing: false, detail: timedOut ? `timed out after ${CHECK_TIMEOUT_MS / 1000}s\n${detail}` : detail };
	}
}

function render(items: Item[]): string {
	if (items.length === 0) return "no checks bound yet";
	return items
		.map((it, i) => `${i + 1}. [${it.passing ? "x" : " "}] ${it.claim}\n     ${it.command}`)
		.join("\n");
}

export function createVerifyToolDefinition(
	cwd: string,
	options?: VerifyToolOptions,
): ToolDefinition<typeof verifySchema, VerifyToolDetails> {
	const runner = options?.runner ?? defaultRunner;
	// Per-session: the checklist describes the task in front of the agent.
	let items: Item[] = [];

	return {
		name: "verify",
		label: "verify",
		description:
			"Bind a claim about your work to a shell command that proves it, then re-run them all before you finish. " +
			"Use action=check with a claim and a command that exits non-zero when the claim does not hold -- one per " +
			"stated requirement at minimum, and one per case a requirement is graded on: the ordinary input, the empty " +
			"and boundary ones, the numeric tolerance the task states, the rejection path and its exit code, and " +
			"anything that must NOT have changed. Use action=submit before finishing; it re-runs every check, which is " +
			"what catches a requirement met earlier and quietly broken since.",
		promptSnippet: verifyToolSystemPromptContribution.snippet,
		parameters: verifySchema,
		async execute(
			_toolCallId,
			{ action, claim, command }: VerifyToolInput,
			_signal?: AbortSignal,
			_onUpdate?,
			ctx?: ExtensionContext,
		) {
			const dir = ctx?.cwd || cwd;

			if (action === "list") {
				return {
					content: [{ type: "text" as const, text: render(items) }],
					details: { items: [...items], failing: items.filter((i) => !i.passing).length },
				};
			}

			if (action === "check") {
				if (!claim || !command) {
					return {
						content: [
							{
								type: "text" as const,
								text: "action=check needs both a claim and a command that exits non-zero when it does not hold.",
							},
						],
						details: { items: [...items], failing: items.filter((i) => !i.passing).length },
						isError: true,
					};
				}
				const { passing, detail } = await runner(command, dir);
				items = [...items.filter((it) => it.claim !== claim), { claim, command, passing, detail }];
				const failing = items.filter((it) => !it.passing).length;
				return {
					content: [
						{
							type: "text" as const,
							text:
								`${passing ? "PASS" : "FAIL"} ${claim}\n` +
								(detail ? `${detail}\n` : "") +
								`\n${render(items)}\n\n${failing} of ${items.length} still failing`,
						},
					],
					details: { items: [...items], failing },
				};
			}

			// submit
			if (items.length === 0) {
				return {
					content: [
						{
							type: "text" as const,
							text:
								"No checks bound, so submit has nothing to re-run and cannot tell you the work is right. " +
								"Bind one per stated requirement with action=check first.",
						},
					],
					details: { items: [], failing: 0 },
					isError: true,
				};
			}
			const rechecked: Item[] = [];
			for (const it of items) {
				const { passing, detail } = await runner(it.command, dir);
				rechecked.push({ ...it, passing, detail });
			}
			items = rechecked;
			const failed = items.filter((it) => !it.passing);
			if (failed.length > 0) {
				return {
					content: [
						{
							type: "text" as const,
							text:
								`${failed.length} of ${items.length} checks FAIL on re-run.\n\n${render(items)}\n\n` +
								failed.map((f) => `- ${f.claim}\n  ${f.detail}`).join("\n"),
						},
					],
					details: { items: [...items], failing: failed.length },
					isError: true,
				};
			}
			return {
				content: [
					{
						type: "text" as const,
						text:
							`All ${items.length} checks pass on re-run.\n\n${render(items)}\n\n` +
							"Note what this does and does not say. It says the claims you bound hold right now. It does " +
							"not say they cover what is graded: measured on this setup, graders ran about six times as " +
							"many checks as the agent had bound, and every failure was inside that gap. Before " +
							"finishing, name any behaviour the task asks for that has no check above.",
					},
				],
				details: { items: [...items], failing: 0 },
			};
		},
	};
}

export function formatVerifyCall(args: VerifyToolInput | undefined, theme: Theme): string {
	const action = args?.action ?? "?";
	let text = `${theme.fg("toolTitle", theme.bold("verify"))} ${theme.fg("toolOutput", action)}`;
	if (args?.claim) text += ` ${theme.fg("muted", args.claim.slice(0, 60))}`;
	return text;
}

export function formatVerifyResult(
	result: { content: Array<{ type: string; text?: string }>; details?: VerifyToolDetails },
	_options: ToolRenderResultOptions,
	theme: Theme,
	showImages: boolean,
): string {
	const output = getTextOutput(result, showImages).trim();
	if (!output) return "";
	const failing = result.details?.failing ?? 0;
	const colour = failing > 0 ? "warning" : "toolOutput";
	return `\n${output
		.split("\n")
		.map((line) => theme.fg(colour, line))
		.join("\n")}`;
}

export function createVerifyTool(cwd: string, options?: VerifyToolOptions): AgentTool<typeof verifySchema> {
	return createVerifyToolDefinition(cwd, options) as unknown as AgentTool<typeof verifySchema>;
}
