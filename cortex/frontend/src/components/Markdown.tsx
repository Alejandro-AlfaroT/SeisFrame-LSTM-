import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const WIKILINK = /!?\[\[([^\]|#]*)(#[^\]|]*)?(?:\|([^\]]*))?\]\]/g;
export const REF_PREFIX = "cortex:ref/";

/** Obsidian-style [[target#heading|alias]] -> a link the viewer resolves to a record. */
export function wikilinksToLinks(md: string): string {
  return md.replace(WIKILINK, (_m, target: string, heading: string | undefined, alias: string | undefined) => {
    const ref = `${target}${heading ?? ""}`.trim();
    const label = (alias || target.split("/").pop() || ref).replace(/[[\]]/g, "");
    return `[${label}](${REF_PREFIX}${encodeURIComponent(ref)})`;
  });
}

/** Imported notes are untrusted: only http(s)/mailto leave the app; relative paths become record refs. */
export function safeUrl(url: string): string {
  if (url.startsWith(REF_PREFIX)) return url;
  if (url.startsWith("#")) return "";
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) {
    return /^(https?|mailto):/i.test(url) ? defaultUrlTransform(url) : "";
  }
  return REF_PREFIX + encodeURIComponent(decodeURIComponentSafe(url));
}

function decodeURIComponentSafe(text: string): string {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

export default function Markdown({ text, onOpenRef }: { text: string; onOpenRef: (ref: string) => void }) {
  const components: Components = {
    a: ({ href, children }) => {
      if (href?.startsWith(REF_PREFIX)) {
        const ref = decodeURIComponentSafe(href.slice(REF_PREFIX.length));
        return (
          <a
            href="#"
            className="wikilink"
            title={ref}
            onClick={(e) => {
              e.preventDefault();
              onOpenRef(ref);
            }}
          >
            {children}
          </a>
        );
      }
      if (!href) return <span>{children}</span>;
      return (
        <a href={href} target="_blank" rel="noopener noreferrer nofollow">
          {children}
        </a>
      );
    },
    // Never fetch remote images from imported notes (tracking pixels, private-network probes).
    img: ({ alt }) => <span className="img-blocked">[image not loaded{alt ? `: ${alt}` : ""}]</span>,
  };
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={safeUrl} components={components}>
        {wikilinksToLinks(text)}
      </ReactMarkdown>
    </div>
  );
}
