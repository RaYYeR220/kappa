import { Fragment, type ReactNode } from "react";

/** Inline markdown: **bold**, *italic*, `code`. Enough for the pre-registration. */
function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) out.push(<strong key={k++}>{t.slice(2, -2)}</strong>);
    else if (t.startsWith("`")) out.push(<code key={k++}>{t.slice(1, -1)}</code>);
    else out.push(<em key={k++}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export interface MdSection {
  title: string;
  body: string;
}

/** Split a document into its level-2 sections; the text before the first one is the lede. */
export function sections(md: string): { lede: string; sections: MdSection[] } {
  const parts = md.split(/^## /m);
  const head = parts.shift() ?? "";
  const lede = head.replace(/^# .*$/m, "").trim();
  return {
    lede,
    sections: parts.map((p) => {
      const nl = p.indexOf("\n");
      return { title: p.slice(0, nl).trim(), body: p.slice(nl + 1).trim() };
    }),
  };
}

/** Block markdown: paragraphs, "-" lists and "1." lists. A paragraph starting with `mark` is flagged. */
export function Markdown({ source, mark }: { source: string; mark?: string }) {
  const blocks = source.split(/\n{2,}/);
  return (
    <>
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        if (lines.every((l) => /^- /.test(l)))
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{inline(l.slice(2))}</li>
              ))}
            </ul>
          );
        if (lines.every((l) => /^\d+\. /.test(l)))
          return (
            <ol key={i}>
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^\d+\. /, ""))}</li>
              ))}
            </ol>
          );
        return (
          <p key={i} data-mark={mark && block.startsWith(mark) ? "" : undefined}>
            {lines.map((l, j) => (
              <Fragment key={j}>
                {j > 0 && " "}
                {inline(l)}
              </Fragment>
            ))}
          </p>
        );
      })}
    </>
  );
}
