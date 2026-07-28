import type { CronJob, CronJobMutation } from "./api";

export interface DeliverEntry {
  gateway: string;
  channel: string;
}

export interface CronJobFormState {
  name: string;
  prompt: string;
  schedule: string;
  deliver_entries: DeliverEntry[];
  skills: string[];
  provider: string;
  model: string;
  base_url: string;
  script: string;
  no_agent: boolean;
  context_from: string;
  enabled_toolsets: string[];
  workdir: string;
  model_fallback_type: '' | 'model_switch' | 'retry_interval';
  model_fallback_models: string[];
  model_fallback_interval: number;
}

/** Split a comma/newline list (or array) into trimmed, non-empty items. */
export function splitCronList(value: unknown): string[] {
  const items = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(/[\n,]/)
      : [];
  return items.map((item) => String(item).trim()).filter(Boolean);
}

/** Trim to a non-empty string, or null. Optionally strip trailing slashes
 * (base URLs). Mirrors the backend's `_cron_optional_text`. */
function optionalText(value: string, stripTrailingSlash = false): string | null {
  const text = stripTrailingSlash ? value.trim().replace(/\/+$/, "") : value.trim();
  return text || null;
}

/** Coerce a stored list/string field back into the textarea's newline form. */
function listToText(value: unknown): string {
  if (Array.isArray(value)) return splitCronList(value).join("\n");
  return typeof value === "string" ? value : "";
}

/** Read a stored string field as a plain string ("" when absent). */
function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** Build the create/update payload. Optional fields collapse to null so an
 * update explicitly clears them rather than leaving stale values. */
export function buildCronJobPayload(form: CronJobFormState): CronJobMutation {
  const contextFrom = splitCronList(form.context_from);
  const enabledToolsets = form.enabled_toolsets.filter(Boolean);
  const entries = form.deliver_entries?.length ? form.deliver_entries : [{ gateway: "local", channel: "" }];
  const deliver = entries
    .map(e => {
      const g = e.gateway.trim() || "local";
      const c = e.channel?.trim() || "";
      return c ? `${g}:${c}` : g;
    })
    .join(",");
  return {
    name: form.name.trim(),
    prompt: form.prompt.trim(),
    schedule: form.schedule.trim(),
    deliver,
    skills: form.skills.filter(Boolean),
    provider: optionalText(form.provider),
    model: optionalText(form.model),
    base_url: optionalText(form.base_url, true),
    script: optionalText(form.script),
    no_agent: Boolean(form.no_agent),
    context_from: contextFrom.length > 0 ? contextFrom : null,
    enabled_toolsets: enabledToolsets.length > 0 ? enabledToolsets : null,
    workdir: optionalText(form.workdir),
    model_fallback: form.no_agent || !form.model_fallback_type ? null : {
      type: form.model_fallback_type,
      models: form.model_fallback_type === 'model_switch' ? form.model_fallback_models.filter(Boolean) : [],
      retry_interval_seconds: form.model_fallback_type === 'retry_interval' ? Number(form.model_fallback_interval) : undefined,
    },
  };
}

export function cronJobHasExecutionContent(
  job: Pick<CronJobMutation, "prompt" | "skills" | "script">,
): boolean {
  const skills = Array.isArray(job.skills) ? job.skills.filter(Boolean) : [];
  return Boolean(asString(job.prompt).trim() || asString(job.script).trim() || skills.length);
}

export function cronJobFormFromJob(job: CronJob): CronJobFormState {
  const deliverStr = asString(job.deliver).trim();
  const deliver_entries: DeliverEntry[] = deliverStr
    ? deliverStr.split(",").map((part) => {
        const colonIdx = part.indexOf(":");
        if (colonIdx === -1) {
          return { gateway: part.trim(), channel: "" };
        } else {
          return {
            gateway: part.substring(0, colonIdx).trim(),
            channel: part.substring(colonIdx + 1).trim(),
          };
        }
      })
    : [{ gateway: "local", channel: "" }];

  return {
    name: asString(job.name),
    prompt: asString(job.prompt),
    schedule:
      asString(job.schedule?.expr) ||
      asString(job.schedule?.run_at) ||
      asString(job.schedule_display),
    deliver_entries,
    skills: Array.isArray(job.skills) ? job.skills.filter(Boolean) : [],
    provider: asString(job.provider),
    model: asString(job.model),
    base_url: asString(job.base_url),
    script: asString(job.script),
    no_agent: Boolean(job.no_agent),
    context_from: listToText(job.context_from),
    enabled_toolsets: splitCronList(job.enabled_toolsets),
    workdir: asString(job.workdir),
    model_fallback_type: (job.model_fallback?.type as '' | 'model_switch' | 'retry_interval') || '',
    model_fallback_models: job.model_fallback?.models || [],
    model_fallback_interval: job.model_fallback?.retry_interval_seconds || 300,
  };
}
