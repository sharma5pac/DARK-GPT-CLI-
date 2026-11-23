import React, { useEffect, useRef } from "react";

export const ACCEPT =
  ".pdf,.txt,.md,.markdown,.csv,.tsv,.json,.log,.xml,.yaml,.yml,.ini,.conf,.cfg," +
  ".py,.js,.ts,.html,.htm,.sh,.sql,.c,.h,.java,.go,.png,.jpg,.jpeg,.webp,.gif,.bmp";

function AttachmentChip({ item, onRemove }) {
  return (
    <div className={`chip ${item.kind === "image" ? "chip-image" : ""}`}>
      {item.kind === "image" ? <img src={item.previewUrl} alt={item.name} /> : <span className="chip-icon">📄</span>}
      <span className="chip-name" title={item.name}>{item.name}</span>
      <button type="button" className="chip-remove" onClick={() => onRemove(item.id)} aria-label={`Remove ${item.name}`}>✕</button>
    </div>
  );
}

export default function Composer({ value, setValue, attachments, addFiles, removeAttachment, onSend, onStop, streaming, visionWarning, webSearch, setWebSearch }) {
  const areaRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => { if (streaming) return; if (!value.trim() && attachments.length === 0) return; onSend(); };
  const onKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } };
  const onPaste = (e) => { const files = Array.from(e.clipboardData?.files || []); if (files.length > 0) { e.preventDefault(); addFiles(files); } };
  const onDrop = (e) => { e.preventDefault(); addFiles(Array.from(e.dataTransfer?.files || [])); };

  return (
    <div className="composer-wrap">
      <div className={`composer ${attachments.length ? "has-files" : ""}`} onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
        {attachments.length > 0 && <div className="chips">{attachments.map((item) => <AttachmentChip key={item.id} item={item} onRemove={removeAttachment} />)}</div>}
        <div className="composer-row">
          <button type="button" className="icon-btn" title="Attach files" aria-label="Attach files" onClick={() => fileRef.current?.click()}>
            <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true"><path fill="currentColor" d="M16.5 6.5v8.25a4.5 4.5 0 0 1-9 0V6a3 3 0 0 1 6 0v8.25a1.5 1.5 0 0 1-3 0V7h-1.5v7.25a3 3 0 0 0 6 0V6a4.5 4.5 0 0 0-9 0v8.75a6 6 0 0 0 12 0V6.5h-1.5Z"/></svg>
          </button>
          <button type="button" className={`icon-btn ${webSearch ? "active" : ""}`} title={webSearch ? "Web search on" : "Web search off"} aria-label="Toggle web search" aria-pressed={webSearch} onClick={() => setWebSearch(v => !v)}>
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><circle cx="12" cy="12" r="9"/><path d="M3.6 9h16.8M3.6 15h16.8"/><path d="M12 3c2.5 2.6 4 5.8 4 9s-1.5 6.4-4 9c-2.5-2.6-4-5.8-4-9s1.5-6.4 4-9z"/></svg>
          </button>
          <textarea ref={areaRef} className="composer-input" rows={1} placeholder="Message DarkGPT…" value={value} onChange={(e) => setValue(e.target.value)} onKeyDown={onKeyDown} onPaste={onPaste} aria-label="Message DarkGPT" />
          {streaming ? (
            <button type="button" className="send-btn stop" onClick={onStop} title="Stop generating" aria-label="Stop generating"><span className="stop-square" /></button>
          ) : (
            <button type="button" className="send-btn" onClick={submit} disabled={!value.trim() && attachments.length === 0} title="Send" aria-label="Send">
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M12 4l7 8h-4.5v8h-5v-8H5l7-8Z" /></svg>
            </button>
          )}
        </div>
      </div>
      <input ref={fileRef} type="file" multiple accept={ACCEPT} className="hidden-file" onChange={(e) => { addFiles(Array.from(e.target.files || [])); e.target.value = ""; }} />
      {visionWarning && <p className="composer-note warn">This model cannot see images — pick one marked “images” in the sidebar, or the picture will be ignored.</p>}
      <p className="composer-note">{webSearch ? "Web search on · " : ""}Paste, drop or attach files. Enter sends, Shift+Enter adds a line.</p>
    </div>
  );
}
