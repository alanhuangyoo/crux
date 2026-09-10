/**
 * crux-verify — the bound-checklist mechanism, as a pi extension.
 *
 * The measurement this exists for: across eight graded tasks the suites ran 206
 * tests, the agent passed 176 of them, and scored on none -- every task failed
 * one to three tests of its own set, 94 of 97 on one. Reading the transcripts,
 * the agent had bound 35 checks of its own against those 206 tests: five for the
 * 97-test suite, one for a thirteen. It then reported "all 3 item(s) verified"
 * and stopped.
 *
 * A prompt asking for more checks does not fix that, because from inside the
 * turn every individual check still looks sufficient. What changes the outcome
 * is a mechanism: a claim is only closed by a command that re-runs and passes,
 * and finishing re-runs all of them. That is what caught, in the earlier Python
 * toolkit, the requirement that was met at step 10 and quietly broken at step 40.
 *
 * Ported here rather than left in crux's own scaffold so the thing measured on
 * the benchmark and the thing used in the terminal are the same code.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const run = promisify(execFile);

// Long enough for a build or a test suite, short enough that a hung check does
// not eat the turn. A verification command that needs longer than this is doing
// something other than verifying.
const CHECK_TIMEOUT_MS = 120_000;

type Item = {
  claim: string;
  verify: string;
  passing: boolean;
  detail: string;
};

type Outcome = { passing: boolean; detail: string };

async function evaluate(command: string, cwd: string): Promise<Outcome> {
  try {
    const { stdout, stderr } = await run("bash", ["-lc", command], {
      cwd,
      timeout: CHECK_TIMEOUT_MS,
      maxBuffer: 1024 * 1024,
    });
    const tail = (stdout + stderr).trim().slice(-400);
    return { passing: true, detail: tail };
  } catch (err: any) {
    // A non-zero exit is the signal, not an accident: the check is written so
    // that failure means the claim does not hold.
    const tail = [err?.stdout, err?.stderr, err?.message]
      .filter(Boolean)
      .join("\n")
      .trim()
      .slice(-400);
    const timedOut = err?.killed === true || err?.signal === "SIGTERM";
    return {
      passing: false,
      detail: timedOut ? `timed out after ${CHECK_TIMEOUT_MS / 1000}s\n${tail}` : tail,
    };
  }
}

function render(items: Item[]): string {
  if (items.length === 0) return "no checks bound yet";
  return items
    .map((it, i) => `${i + 1}. [${it.passing ? "x" : " "}] ${it.claim}\n     check: ${it.verify}`)
    .join("\n");
}

export default function (pi: ExtensionAPI) {
  let items: Item[] = [];

  // Rebuilt from the session so a branch or a resume does not silently start
  // with an empty checklist and let `crux_submit` pass on nothing.
  pi.on("session_start", async (_event, ctx) => {
    items = [];
    for (const entry of ctx.sessionManager.getBranch()) {
      if (
        entry.type === "message" &&
        (entry as any).message?.role === "toolResult" &&
        ["crux_check", "crux_submit"].includes((entry as any).message?.toolName)
      ) {
        items = (entry as any).message?.details?.items ?? items;
      }
    }
  });

  pi.registerTool({
    name: "crux_check",
    label: "Bind a check",
    description:
      "Bind a claim about the work to a shell command that proves it, and run " +
      "that command now. Use one per stated requirement at minimum, and one per " +
      "case a requirement is graded on: the ordinary input, the empty and " +
      "boundary ones, the stated numeric tolerance, the rejection path and its " +
      "exit code, and anything that must NOT have changed. The command must " +
      "exit non-zero when the claim does not hold.",
    parameters: Type.Object({
      claim: Type.String({ description: "What must be true, in one line" }),
      verify: Type.String({
        description: "Shell command that exits 0 only if the claim holds",
      }),
    }),
    async execute(_id, params, _signal, _onUpdate, ctx) {
      const { passing, detail } = await evaluate(params.verify, ctx.cwd ?? process.cwd());
      items = [
        ...items.filter((it) => it.claim !== params.claim),
        { claim: params.claim, verify: params.verify, passing, detail },
      ];
      const open = items.filter((it) => !it.passing).length;
      return {
        content: [
          {
            type: "text",
            text:
              `${passing ? "PASS" : "FAIL"} ${params.claim}\n` +
              (detail ? `${detail}\n` : "") +
              `\n${render(items)}\n\n${open} of ${items.length} still failing`,
          },
        ],
        details: { items: [...items] },
      };
    },
  });

  pi.registerTool({
    name: "crux_submit",
    label: "Re-run every check",
    description:
      "Re-run every bound check and report the result. Run this before you " +
      "finish. It exists because a requirement met earlier and quietly broken " +
      "since is the most common way a task fails while looking done.",
    parameters: Type.Object({}),
    async execute(_id, _params, _signal, _onUpdate, ctx) {
      if (items.length === 0) {
        return {
          content: [
            {
              type: "text",
              text:
                "No checks bound. crux_submit has nothing to re-run, so it " +
                "cannot tell you the work is right. Bind one per stated " +
                "requirement with crux_check first.",
            },
          ],
          details: { items: [] },
          isError: true,
        };
      }
      const cwd = ctx.cwd ?? process.cwd();
      const rechecked: Item[] = [];
      for (const it of items) {
        const { passing, detail } = await evaluate(it.verify, cwd);
        rechecked.push({ ...it, passing, detail });
      }
      items = rechecked;
      const failing = items.filter((it) => !it.passing);
      const body = render(items);
      if (failing.length > 0) {
        return {
          content: [
            {
              type: "text",
              text:
                `${failing.length} of ${items.length} checks FAIL on re-run.\n\n${body}\n\n` +
                failing.map((f) => `- ${f.claim}\n  ${f.detail}`).join("\n"),
            },
          ],
          details: { items: [...items] },
          isError: true,
        };
      }
      return {
        content: [
          {
            type: "text",
            text:
              `All ${items.length} checks pass on re-run.\n\n${body}\n\n` +
              "Note what this does and does not say: it says the claims you " +
              "bound hold right now. On the measured runs the graders ran " +
              "roughly six times as many checks as the agent had bound, and " +
              "every failure was inside that gap. Before finishing, name any " +
              "behaviour the task asks for that has no check above.",
          },
        ],
        details: { items: [...items] },
      };
    },
  });

  pi.registerCommand("checks", {
    description: "Show the bound checks and their last result",
    handler: async (_args, ctx) => {
      ctx.ui.notify(render(items), items.some((i) => !i.passing) ? "warning" : "info");
    },
  });
}
