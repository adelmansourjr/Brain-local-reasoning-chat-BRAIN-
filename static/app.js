// Elements
const elMsgs = document.getElementById('messages');
const elInput = document.getElementById('input');
const elForm = document.getElementById('composer');
const elSend = document.getElementById('send');

const elUseWeb = document.getElementById('useWeb');
const elUseReasoning = document.getElementById('useReasoning');
const elTemp = document.getElementById('temp');
const elTopP = document.getElementById('topp');
const elMaxNew = document.getElementById('maxNew');
const elRegion = document.getElementById('region');

const elConvList = document.getElementById('convList');
const elNewChat = document.getElementById('newChat');

let ws = null;

// ---- Conversations (localStorage) --------------------------------------------
const LS_KEY = 'lr_conversations_v1';

function loadConvs(){
  try{ return JSON.parse(localStorage.getItem(LS_KEY) || '[]'); }catch{ return []; }
}
function saveConvs(convs){ localStorage.setItem(LS_KEY, JSON.stringify(convs)); }
function titleFrom(text){ return (text || 'New chat').slice(0, 42); }

let conversations = loadConvs();
let activeId = conversations[0]?.id || null;
function newId(){ return Math.random().toString(36).slice(2, 10); }

function setActive(id){
  activeId = id;
  renderConvList();
  renderMessages();
}

function addConversation(firstUserMsg=''){
  const id = newId();
  const conv = { id, title: titleFrom(firstUserMsg), msgs: [] };
  conversations.unshift(conv);
  saveConvs(conversations);
  setActive(id);
}
function deleteConv(id){
  const idx = conversations.findIndex(c=>c.id===id);
  if (idx>=0){ conversations.splice(idx,1); saveConvs(conversations); }
  if (activeId===id){ activeId = conversations[0]?.id || null; }
  renderConvList(); renderMessages();
}
function activeConv(){
  return conversations.find(c=>c.id===activeId) || null;
}
function pushMsg(role, text, sources){
  let conv = activeConv();
  if (!conv){ addConversation(); conv = activeConv(); }
  conv.msgs.push({ role, text, ts: Date.now(), sources: sources||[] });
  if (role==='user' && (!conv.title || conv.title==='New chat')) conv.title = titleFrom(text);
  saveConvs(conversations);
}

// ---- UI: conversations --------------------------------------------------------
function renderConvList(){
  elConvList.innerHTML = '';
  conversations.forEach(c=>{
    const row = document.createElement('div');
    row.className = 'conv' + (c.id===activeId?' active':'');
    row.onclick = ()=> setActive(c.id);
    const title = document.createElement('div'); title.className='title'; title.textContent = c.title || 'New chat';
    const del = document.createElement('button'); del.className='del'; del.textContent='×'; del.title='Delete';
    del.onclick = (e)=>{ e.stopPropagation(); deleteConv(c.id); };
    row.appendChild(title); row.appendChild(del);
    elConvList.appendChild(row);
  });
}
elNewChat.onclick = ()=>{ addConversation(); };

// ---- Messages UI --------------------------------------------------------------
function addMessage(role, text){
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role==='user' ? 'U' : 'A';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  wrap.appendChild(avatar); wrap.appendChild(bubble);
  elMsgs.appendChild(wrap);
  elMsgs.parentElement.scrollTop = elMsgs.parentElement.scrollHeight;
  return { wrap, bubble };
}
function addSources(bubble, sources){
  if (!sources || !sources.length) return;
  const s = document.createElement('div');
  s.className = 'sources';
  s.innerHTML = sources.map((x,i)=>`${i+1}. <a target="_blank" rel="noopener" href="${x.url}">${x.title}</a>`).join('<br/>');
  bubble.appendChild(s);
}
function addCopyButton(bubble, text){
  const btn = document.createElement('button');
  btn.className = 'copy';
  btn.textContent = 'Copy';
  btn.onclick = async ()=> {
    try { await navigator.clipboard.writeText(text); btn.textContent = 'Copied ✓'; setTimeout(()=>btn.textContent='Copy', 1200); }
    catch { btn.textContent = 'Error'; setTimeout(()=>btn.textContent='Copy', 1200); }
  };
  bubble.appendChild(btn);
}

function renderMessages(){
  elMsgs.innerHTML = '';
  const conv = activeConv();
  if (!conv) return;
  conv.msgs.forEach(m=>{
    const { bubble } = addMessage(m.role, m.text);
    if (m.role==='assistant'){
      addSources(bubble, m.sources);
      addCopyButton(bubble, m.text);
    }
  });
}
renderConvList(); renderMessages();

// ---- Composer / WebSocket -----------------------------------------------------
function setBusy(b){ elSend.disabled = b; elInput.disabled = b; }

function openWS(){
  if (ws && ws.readyState===WebSocket.OPEN) return ws;
  const url = (location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws';
  ws = new WebSocket(url);
  return ws;
}

function sendPrompt(message){
  const payload = {
    message,
    use_web: elUseWeb.checked,
    use_reasoning: elUseReasoning.checked,
    temperature: parseFloat(elTemp.value),
    top_p: parseFloat(elTopP.value),
    max_new_tokens: parseInt(elMaxNew.value,10),
    region: elRegion.value || 'uk-en',
    k: 3, reflect: true,
  };
  const sock = openWS();
  sock.onopen = ()=> sock.send(JSON.stringify(payload));
  return sock;
}

elForm.addEventListener('submit', (e)=>{
  e.preventDefault();
  const message = elInput.value.trim();
  if (!message) return;

  // ensure active convo
  if (!activeConv()) addConversation(message);
  else if (activeConv().msgs.length===0) { activeConv().title = titleFrom(message); saveConvs(conversations); renderConvList(); }

  // UI
  const { bubble: userBubble } = addMessage('user', message);
  pushMsg('user', message);
  elInput.value = ''; elInput.style.height = '42px';
  setBusy(true);

  const { bubble: asstBubble } = addMessage('assistant', '');
  let liveText = '';

  try{
    const sock = sendPrompt(message);
    sock.onmessage = (ev)=>{
      try{
        const msg = JSON.parse(ev.data);
        if (msg.type==='status'){
          asstBubble.textContent = `[${msg.data}]`;
        } else if (msg.type==='start'){
          asstBubble.textContent = ''; liveText = '';
        } else if (msg.type==='answer'){
          asstBubble.textContent += msg.delta; liveText += msg.delta;
          elMsgs.parentElement.scrollTop = elMsgs.parentElement.scrollHeight;
        } else if (msg.type==='done'){
          if (msg.sources && msg.sources.length) addSources(asstBubble, msg.sources);
          addCopyButton(asstBubble, liveText);
          pushMsg('assistant', liveText, msg.sources || []);
          setBusy(false); sock.close();
        } else if (msg.type==='error'){
          asstBubble.textContent = 'Error: ' + msg.data;
          setBusy(false); sock.close();
        }
      }catch{}
    };
    sock.onerror = ()=>{
      asstBubble.textContent = 'Connection error.';
      setBusy(false);
    };
  }catch(err){
    asstBubble.textContent = `Error: ${err}`;
    setBusy(false);
  }
});

// Auto-expand textarea (Shift+Enter = newline; Enter = send)
elInput.addEventListener('input', ()=>{
  elInput.style.height = '42px';
  elInput.style.height = Math.min(elInput.scrollHeight, 260) + 'px';
});
elInput.addEventListener('keydown', (e)=>{
  if (e.key==='Enter' && !e.shiftKey){
    e.preventDefault();
    elForm.requestSubmit();
  }
});