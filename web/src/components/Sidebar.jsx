import React from "react";

function ModelPicker({ models, model, setModel, includeUncensored, setIncludeUncensored }) {
  const free = models?.free || [];
  const paid = includeUncensored ? models?.uncensored || [] : [];
  return (
    <div className="model-box">
      <label className="model-label" htmlFor="model-select">Model</label>
      <select id="model-select" className="model-select" value={model} onChange={(e) => setModel(e.target.value)}>
        <optgroup label={`Free (${free.length})`}>
          {free.map((entry) => (
            <option key={entry.id} value={entry.id}>{entry.id}{entry.vision ? "  ·  images" : ""}{entry.context ? `  ·  ${Math.round(entry.context / 1000)}k` : ""}</option>
          ))}
        </optgroup>
        {paid.length > 0 && (
          <optgroup label={`Uncensored (${paid.length}) · paid`}>
            {paid.map((entry) => (
              <option key={entry.id} value={entry.id}>{entry.id} · ${entry.price_in}/${entry.price_out} per M{entry.vision ? " · images" : ""}</option>
            ))}
          </optgroup>
        )}
      </select>
      <label className="switch-row">
        <input type="checkbox" checked={includeUncensored} onChange={(e) => setIncludeUncensored(e.target.checked)} />
        <span>Uncensored models</span>
      </label>
      <p className="model-note">Uncensored by design (Dolphin, TheDrummer, Euryale). None are free — billed per token against your OpenRouter credit.</p>
    </div>
  );
}

export default function Sidebar({ conversations, activeId, onNew, onOpen, onDelete, models, model, setModel, includeUncensored, setIncludeUncensored, keyPresent, open, onClose }) {
  return (
    <aside className={`sidebar ${open ? "is-open" : ""}`}>
      <div className="sidebar-top">
        <div className="brand"><span className="brand-mark">◍</span> DarkGPT</div>
        <button className="new-chat" onClick={onNew} title="New chat"><span className="plus">＋</span> New chat</button>
      </div>
      <div className="conv-list" role="list">
        {conversations.length === 0 && <p className="conv-empty">No chats yet. Start one below.</p>}
        {conversations.map((conv) => (
          <div key={conv.id} role="listitem" className={`conv-row ${conv.id === activeId ? "active" : ""}`}>
            <button className="conv-open" onClick={() => onOpen(conv.id)} title={conv.title}>{conv.title}</button>
            <button className="conv-del" onClick={(e) => { e.stopPropagation(); onDelete(conv.id); }} title="Delete chat" aria-label={`Delete ${conv.title}`}>✕</button>
          </div>
        ))}
      </div>
      <div className="sidebar-bottom">
        <ModelPicker models={models} model={model} setModel={setModel} includeUncensored={includeUncensored} setIncludeUncensored={setIncludeUncensored} />
        <div className="key-row"><span className={keyPresent ? "dot ok" : "dot bad"} />{keyPresent ? "OpenRouter key loaded" : "No OpenRouter key — add one to .env"}</div>
      </div>
      <button className="sidebar-close" onClick={onClose} aria-label="Close sidebar">✕</button>
    </aside>
  );
}
