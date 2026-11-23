const json = async (res) => {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return res.json();
};

export const getModels = () => fetch("/api/models").then(json);
export const getConversations = () => fetch("/api/conversations").then(json);

export const getMessages = (id) =>
  fetch(`/api/conversations/${id}/messages`).then(json);

export const createConversation = (title, model) =>
  fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, model }),
  }).then(json);

export const deleteConversation = (id) =>
  fetch(`/api/conversations/${id}`, { method: "DELETE" }).then(json);

export const deleteMessage = (id) =>
  fetch(`/api/messages/${id}/delete`, { method: "POST" }).then(json);

/**
 * POST /api/chat and consume the SSE stream.
 * onEvent receives {type: 'meta'|'delta'|'done'|'error'|'warning', ...}
 */
async function consumeSSE(res, onEvent) {
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try { const body = await res.json(); detail = body.detail || detail; } catch {}
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try { onEvent(JSON.parse(line.slice(5).trim())); } catch {}
    }
  }
}

export async function streamChat({ message, model, conversationId, resume, files, webSearch }, onEvent, signal) {
  const form = new FormData();
  form.append("message", message || "");
  form.append("model", model);
  if (conversationId) form.append("conversation_id", String(conversationId));
  form.append("resume", resume ? "true" : "false");
  form.append("web_search_enabled", webSearch ? "true" : "false");
  for (const file of files || []) form.append("files", file, file.name);
  const res = await fetch("/api/chat", { method: "POST", body: form, signal });
  await consumeSSE(res, onEvent);
}

export async function streamAgent({ message, model, conversationId }, onEvent, signal) {
  const form = new FormData();
  form.append("message", message || "");
  form.append("model", model || "nvidia/nemotron-3.5-lightning:free");
  if (conversationId) form.append("conversation_id", String(conversationId));
  const res = await fetch("/api/agent", { method: "POST", body: form, signal });
  await consumeSSE(res, onEvent);
}
