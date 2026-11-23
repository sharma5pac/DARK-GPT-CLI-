import React, { useEffect, useRef, useState } from "react";
import Message from "./Message.jsx";

const SUGGESTIONS = [
  { icon: "🔐", label: "Explain a CVE", text: "Explain how CVE-2021-44228 (Log4Shell) works and what the fix was." },
  { icon: "📄", label: "Summarise a file", text: "Summarise the attached document in five bullet points." },
  { icon: "💻", label: "Review code", text: "Review this code for bugs and security issues, then show a fixed version." },
  { icon: "🧠", label: "Explain simply", text: "Explain how HTTPS actually protects traffic, in plain language." },
];

export default function Chat({ messages, streamText, streaming, onSuggestion, onFiles, title, model }) {
  const scrollRef = useRef(null);
  const stick = useRef(true);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const node = scrollRef.current;
    if (!node || !stick.current) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, streamText]);

  const onScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    stick.current = node.scrollHeight - node.scrollTop - node.clientHeight < 140;
  };

  const empty = messages.length === 0 && !streaming;
  return (
    <div className="chat" ref={scrollRef} onScroll={onScroll}
      onDragOver={(e) => { e.preventDefault(); if (!dragging) setDragging(true); }}
      onDragLeave={(e) => { if (e.currentTarget.contains(e.relatedTarget)) return; setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); onFiles(Array.from(e.dataTransfer?.files || [])); }}
      data-testid="chat-scroll">
      <div className="thread">
        {empty ? (
          <div className="hero">
            <div className="hero-mark">◍</div>
            <h1>How can I help?</h1>
            <p>Ask anything. Paste an image, drop a file or attach a PDF — then keep talking. The whole thread is remembered.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((item) => (
                <button key={item.label} className="suggestion" onClick={() => onSuggestion(item.text)}>
                  <span className="suggestion-icon">{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="thread-head">
              <span className="thread-title">{title}</span>
              <span className="thread-badge">{model}</span>
            </div>
            {messages.map((message) => <Message key={message.key} message={message} />)}
            {streaming && <Message message={{ role: "assistant", content: streamText }} streaming />}
          </>
        )}
      </div>
      {dragging && <div className="drop-overlay"><div className="drop-card">Drop files to attach</div></div>}
    </div>
  );
}
