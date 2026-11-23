import React, { useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function CodeBlock({ children }) {
  const ref = useRef(null);
  const copy = async () => {
    const text = ref.current?.innerText || "";
    try { await navigator.clipboard.writeText(text); ref.current?.setAttribute("data-copied", "1"); setTimeout(() => ref.current?.removeAttribute("data-copied"), 1200); } catch {}
  };
  return (
    <div className="code-wrap">
      <button className="code-copy" onClick={copy} type="button">Copy</button>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}

const components = { pre: CodeBlock, a: (props) => <a {...props} target="_blank" rel="noreferrer" /> };

function Attachments({ items }) {
  if (!items?.length) return null;
  const images = items.filter((i) => i.kind === "image" && i.url);
  const docs = items.filter((i) => i.kind !== "image" || !i.url);
  return (
    <>
      {images.length > 0 && (
        <div className={`img-grid ${images.length === 1 ? "single" : ""}`}>
          {images.map((item, i) => <a key={`${item.name}-${i}`} href={item.url} target="_blank" rel="noreferrer"><img src={item.url} alt={item.name} loading="lazy" /></a>)}
        </div>
      )}
      {docs.length > 0 && <div className="doc-chips">{docs.map((item, i) => <span className="doc-chip" key={`${item.name}-${i}`}>📄 {item.name}</span>)}</div>}
    </>
  );
}

export default function Message({ message, streaming }) {
  const { role, content, attachments, error } = message;
  if (role === "user") {
    return (
      <div className="msg msg-user" data-role="user">
        <div className="bubble">
          <Attachments items={attachments} />
          {content ? <p className="user-text">{content}</p> : null}
        </div>
      </div>
    );
  }
  return (
    <div className={`msg msg-assistant ${error ? "is-error" : ""}`} data-role="assistant">
      <div className="avatar" aria-hidden="true">◍</div>
      <div className="answer">
        {error ? <div className="error-bubble"><strong>Request failed.</strong> {content}</div>
          : <div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{content || ""}</ReactMarkdown>{streaming && <span className="caret" />}</div>}
      </div>
    </div>
  );
}
