const TYPE_LABELS: Record<string, string> = {
  "single-session-user": "Single session · user",
  "single-session-assistant": "Single session · assistant",
  "multi-session": "Multi-session",
  "knowledge-update": "Knowledge update",
  "temporal-reasoning": "Temporal reasoning",
};

export const typeLabel = (t: string) => TYPE_LABELS[t] ?? t;

/** "2023/05/30 (Tue) 23:40" -> "30 May 2023" */
export function shortDate(s: string): string {
  const m = /(\d{4})\/(\d{1,2})\/(\d{1,2})/.exec(s ?? "");
  if (!m) return s ?? "";
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

/** dataset date -> "YYYY-MM-DD" for <input type="date"> */
export function isoDate(s: string): string {
  const m = /(\d{4})\/(\d{1,2})\/(\d{1,2})/.exec(s ?? "");
  return m ? `${m[1]}-${m[2].padStart(2, "0")}-${m[3].padStart(2, "0")}` : /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : "";
}

export const ms = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${n} ms`);

export const num = (n: number) => n.toLocaleString("en-US");
