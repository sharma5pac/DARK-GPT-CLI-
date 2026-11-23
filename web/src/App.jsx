import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Chat from "./components/Chat.jsx";
import Composer from "./components/Composer.jsx";
import {
  deleteConversation as apiDeleteConversation,
  deleteMessage as apiDeleteMessage,
  getConversations,
  getMessages,
  getModels,
  streamChat,
} from "./api.js";

const DEFAULT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition";
let uid = 0;
const nextKey = () => `local-${++uid}`;

export default function App() {
  const [models, setModels] = useState({ free: [], uncensored: [] });
  const [includeUncensored, setIncludeUncensored] = useState(true);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [title, setTitle] = useState("New chat");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [keyPresent, setKeyPresent] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notice, setNotice] = useState(null);
  const [webSearch, setWebSearch] = useState(false);
  const [sources, setSources] = useState([]);
  const [activity, setActivity] = useState("");
  const [steps, setSteps] = useState([]);
  const [stepSummary, setStepSummary] = useState("");

  const abortRef = useRef(null);
  const freshChat = useRef(false);
  const activeIdRef = useRef(null);

  useEffect(() => { activeIdRef.current = activeId; }, [activeId]);
  useEffect(() => {
    if (!streaming) return undefined;
    const t = setInterval(() => setSteps((s) => s.map((x) => (x.status === "active" ? { ...x, elapsed: (Date.now() - x.startedAt) / 1000 } : x))), 500);
    return () => clearInterval(t);
  }, [streaming]);

  const refreshConversations = useCallback(async () => {
    try { setConversations(await getConversations()); } catch {}
  }, []);

  const loadConversation = useCallback(async (id) => {
    try {
      const rows = await getMessages(id);
      setMessages(rows.map((row) => ({ ...row, key: `m-${row.id}` })));
      setActiveId(id);
      const conv = conversations.find((c) => c.id === id);
      if (conv) setTitle(conv.title);
    } catch (error) { setNotice(`Could not load chat: ${error.message}`); }
  }, [conversations]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [catalogue, rows] = await Promise.all([getModels(), getConversations()]);
        if (cancelled) return;
        setModels(catalogue);
        setKeyPresent(Boolean(catalogue.key_present));
        setConversations(rows);
        if (rows.length > 0 && !freshChat.current) {
          const recent = rows[0];
          setActiveId(recent.id);
          setTitle(recent.title);
          const history = await getMessages(recent.id);
          if (!cancelled) setMessages(history.map((row) => ({ ...row, key: `m-${row.id}` })));
        }
      } catch (error) { if (!cancelled) setNotice(`Backend unreachable: ${error.message}`); }
    })();
    return () => { cancelled = true; };
  }, []);

  const modelEntry = useMemo(() => {
    const all = [...(models.free || []), ...(models.uncensored || [])];
    return all.find((entry) => entry.id === model);
  }, [models, model]);
  const hasImage = attachments.some((item) => item.kind === "image");
  const visionWarning = hasImage && modelEntry && !modelEntry.vision;

  const addFiles = useCallback((files) => {
    if (!files?.length) return;
    setAttachments((current) => {
      const seen = new Set(current.map((item) => `${item.name}:${item.size}`));
      const additions = [];
      for (const file of files) {
        const signature = `${file.name}:${file.size}`;
        if (seen.has(signature)) continue;
        seen.add(signature);
        const kind = file.type.startsWith("image/") ? "image" : "text";
        additions.push({ id: nextKey(), file, name: file.name || "pasted-image.png", size: file.size, kind, previewUrl: kind === "image" ? URL.createObjectURL(file) : null });
      }
      return [...current, ...additions];
    });
  }, []);
  const removeAttachment = useCallback((id) => setAttachments((c) => c.filter((i) => i.id !== id)), []);

  const newChat = useCallback(() => {
    freshChat.current = true;
    setActiveId(null); setMessages([]); setTitle("New chat"); setInput(""); setAttachments([]); setStreamText(""); setNotice(null); setSources([]); setSteps([]); setStepSummary(""); setSidebarOpen(false);
  }, []);

  const openChat = useCallback(async (id) => { if (streaming) return; freshChat.current = true; await loadConversation(id); setSidebarOpen(false); }, [loadConversation, streaming]);
  const removeChat = useCallback(async (id) => {
    await apiDeleteConversation(id);
    setConversations((c) => c.filter((x) => x.id !== id));
    if (id === activeIdRef.current) newChat();
  }, [newChat]);

  const runStream = useCallback(async ({ message, files, resume }) => {
    setStreaming(true); setStreamText(""); setNotice(null); setSources([]); setStepSummary("");
    const started = Date.now();
    const mk = (label, status = "active") => ({ label, status, startedAt: Date.now(), elapsed: 0 });
    let stepList = [mk(webSearch ? "Searching web" : "Thinking")];
    const syncSteps = () => setSteps(stepList.map((s) => ({ ...s, elapsed: s.status === "active" ? (Date.now() - s.startedAt) / 1000 : s.elapsed })));
    const finishLastActive = (label, status = "done") => {
      for (let i = stepList.length - 1; i >= 0; i--) {
        if (stepList[i].status === "active") {
          stepList[i] = { ...stepList[i], status, elapsed: (Date.now() - stepList[i].startedAt) / 1000, label: label || stepList[i].label };
          break;
        }
      }
      syncSteps();
    };
    const pushStep = (label) => { finishLastActive(); stepList = [...stepList, mk(label)]; syncSteps(); };
    setActivity(stepList[0].label + "...");
    syncSteps();

    const controller = new AbortController(); abortRef.current = controller;
    let buffer = ""; let conversationId = activeIdRef.current; let failed = false;
    let sawDelta = false;
    try {
      await streamChat({ message, model, conversationId, resume, files, webSearch }, (event) => {
        if (event.type === "meta") {
          if (event.conversation_id && event.conversation_id !== conversationId) { conversationId = event.conversation_id; setActiveId(conversationId); activeIdRef.current = conversationId; }
          if (event.title) setTitle(event.title);
          if (Array.isArray(event.sources)) setSources(event.sources);
          if (event.sources?.length) { finishLastActive("Searched web", "done"); stepList = [...stepList, mk("Reading sources")]; syncSteps(); setActivity("Reading sources..."); }
        } else if (event.type === "delta") {
          if (!sawDelta) {
            sawDelta = true;
            finishLastActive();
            stepList = [...stepList, mk("Writing response")];
            syncSteps();
          }
          buffer += event.text || "";
          setStreamText(buffer);
          setActivity("Writing...");
          syncSteps();
        } else if (event.type === "warning") { setNotice(event.message); }
        else if (event.type === "error") { failed = true; finishLastActive(event.message?.slice(0, 40) || "Error", "error"); setMessages((c) => [...c, { key: nextKey(), role: "assistant", content: event.message, error: true }]); }
      }, controller.signal);
    } catch (error) {
      if (error.name !== "AbortError") { failed = true; finishLastActive("Failed", "error"); setMessages((c) => [...c, { key: nextKey(), role: "assistant", content: error.message || String(error), error: true }]); }
      else { finishLastActive("Interrupted", "error"); }
    } finally {
      for (let i = 0; i < stepList.length; i++) {
        if (stepList[i].status === "active") stepList[i] = { ...stepList[i], status: failed ? "error" : "done", elapsed: (Date.now() - stepList[i].startedAt) / 1000 };
      }
      const total = ((Date.now() - started) / 1000).toFixed(1);
      const summary = stepList.map((s) => `${s.status === "done" ? "✓" : s.status === "error" ? "✗" : "●"} ${s.label} ${s.elapsed.toFixed(1)}s`).join("  ·  ");
      setStepSummary(`${summary}  ·  total ${total}s`);
      setSteps(stepList.map((s) => ({ ...s })));
      abortRef.current = null; setStreaming(false); setStreamText(""); setActivity("");
      const id = conversationId;
      if (id) { try { const rows = await getMessages(id); setMessages(rows.map((r) => ({ ...r, key: `m-${r.id}` }))); } catch {} await refreshConversations(); }
      if (failed) setNotice("The reply failed. Use Regenerate to try again.");
    }
  }, [model, refreshConversations, webSearch]);

  const send = useCallback(async () => {
    const text = input.trim();
    if ((!text && attachments.length === 0) || streaming) return;
    const files = attachments.map((i) => i.file);
    setMessages((c) => [...c, { key: nextKey(), role: "user", content: text, attachments: attachments.map((i) => ({ name: i.name, kind: i.kind, url: i.previewUrl })) }]);
    setInput(""); setAttachments([]);
    await runStream({ message: text, files, resume: false });
  }, [attachments, input, runStream, streaming]);

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const regenerate = useCallback(async () => {
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (last?.id) { await apiDeleteMessage(last.id); setMessages((c) => c.filter((m) => m.id !== last.id)); }
    await runStream({ message: "", files: [], resume: true });
  }, [messages, runStream]);

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onNew={newChat}
        onOpen={openChat}
        onDelete={removeChat}
        models={models}
        model={model}
        setModel={setModel}
        includeUncensored={includeUncensored}
        setIncludeUncensored={setIncludeUncensored}
        keyPresent={keyPresent}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="main">
        <div className="topbar">
          <button className="menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">☰</button>
          <span className="topbar-title">{title}</span>
          {lastAssistant && !streaming && <button className="regen-btn" onClick={regenerate} title="Regenerate reply">↻ Regenerate</button>}
        </div>
        <Chat messages={messages} streamText={streamText} streaming={streaming} onSuggestion={(text) => setInput(text)} onFiles={addFiles} title={title} model={model} />
        {(steps.length > 0 || stepSummary) && (
          <div className="step-timeline" role="status" aria-live="polite">
            <div className="step-timeline-head">
              <span className="activity-dot" style={{visibility: streaming ? "visible" : "hidden"}} />
              <span>{streaming ? "Agent activity" : "Run summary"}</span>
              {!streaming && stepSummary && <span className="step-total">{stepSummary.split("· total").pop() ? `total ${stepSummary.split("· total").pop().trim()}` : ""}</span>}
            </div>
            <div className="step-list">
              {(steps.length ? steps : []).map((s, i) => {
                const elapsed = s.status === "active" ? ((Date.now() - s.startedAt) / 1000) : s.elapsed;
                return (
                  <div key={`${s.label}-${i}`} className={`step-row step-${s.status}`}>
                    <span className="step-mark">{s.status === "done" ? "✓" : s.status === "error" ? "✗" : "●"}</span>
                    <span className="step-label">{s.label}</span>
                    <span className="step-elapsed">{elapsed.toFixed(1)}s</span>
                  </div>
                );
              })}
            </div>
            {!streaming && stepSummary && <div className="step-summary">{stepSummary}</div>}
          </div>
        )}
        {sources.length > 0 && (
          <div className="notice" role="status" style={{maxWidth:768, margin:"0 auto 6px"}}>
            <span>Sources: {sources.map((s, i) => <a key={i} href={s.href} target="_blank" rel="noreferrer" style={{marginRight:8, color:"inherit"}}>{i+1}. {s.title || s.href}</a>)}</span>
            <button onClick={() => setSources([])} aria-label="Dismiss">✕</button>
          </div>
        )}
        {notice && <div className="notice" role="status">{notice}<button onClick={() => setNotice(null)} aria-label="Dismiss">✕</button></div>}
        <Composer value={input} setValue={setInput} attachments={attachments} addFiles={addFiles} removeAttachment={removeAttachment} onSend={send} onStop={stop} streaming={streaming} visionWarning={visionWarning} webSearch={webSearch} setWebSearch={setWebSearch} />
      </main>
    </div>
  );
}
