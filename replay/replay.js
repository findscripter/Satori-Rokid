(() => {
  'use strict';
  const data = window.REPLAY;
  const get = id => document.getElementById(id);
  const video = get('video'), records = get('records'), scrub = get('scrub');
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const speechById = new Map(data.speech.map(s => [s.id, s]));
  const framesById = new Map(data.frames.map(f => [f.id, f]));
  const icons = {
    play:'M7 4.8 19 12 7 19.2Z', pause:'M8 5v14M16 5v14',
    volume:'M4 9h4l5-4v14l-5-4H4z M16 9a4 4 0 0 1 0 6 M18.5 6a8 8 0 0 1 0 12',
    muted:'M3 9h4l5-4v14l-5-4H3z M16 9l5 6m0-6-5 6',
    expand:'M4 9V4h5m6 0h5v5m0 6v5h-5M9 20H4v-5',
    camera:'M3 7h4l2-3h6l2 3h4v13H3z M15.5 13a3.5 3.5 0 1 1-7 0 3.5 3.5 0 1 1 7 0',
    chat:'M4 5h16v12H9l-5 4z M8 9h8m-8 4h5',
    follow:'M3 5h11M3 10h8M3 15h6m5-3 3 3 4-5',
    calendar:'M4 5h16v15H4z M8 3v4m8-4v4M4 10h16',
    timeline:'M3 18V6m0 12h18M6 14h4m2-4h5m-1 4h5',
    download:'M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4',
    mic:'M9 11V7a3 3 0 0 1 6 0v4a3 3 0 0 1-6 0m-3 0a6 6 0 0 0 12 0m-6 6v3m-4 0h8',
    sound:'M3 10v4m4-8v12m5-15v18m5-13v10m4-8v4',
    info:'M21 12a9 9 0 1 1-18 0 9 9 0 1 1 18 0M12 11v5m0-9v.25',
    close:'M5 5l14 14m0-14L5 19'
  };
  function icon(name) {
    const node = document.createElementNS('http://www.w3.org/2000/svg','svg');
    const path = document.createElementNS(node.namespaceURI,'path');
    for (const [k,v] of Object.entries({viewBox:'0 0 24 24',class:'svg-icon','aria-hidden':'true',fill:'none',stroke:'currentColor','stroke-width':name==='pause'?'3.5':'1.7','stroke-linecap':'round','stroke-linejoin':'round'})) node.setAttribute(k,v);
    path.setAttribute('d',icons[name]); if (name==='play') path.setAttribute('fill','currentColor');
    node.append(path); return node;
  }
  document.querySelectorAll('[data-icon]').forEach(e=>e.replaceChildren(icon(e.dataset.icon)));
  const time = (s, detail=false) => {
    const v=Math.max(0,Math.floor((s+1e-8)*10));
    return String(Math.floor(v/600)).padStart(2,'0')+':'+String(Math.floor(v/10)%60).padStart(2,'0')+(detail?'.'+(v%10):'');
  };
  function el(tag,cls,text) {
    const n=document.createElement(tag); if(cls)n.className=cls;
    if(text!=null)n.textContent=text; return n;
  }
  function button(cls,text,label,action) {
    const n=el('button',cls,text);n.type='button';n.title=label;n.setAttribute('aria-label',label);n.addEventListener('click',action);return n;
  }
  function play() { const p=video.play(); if(p)p.catch(()=>{updatePlayer();}); }
  function seek(at, start=false) {
    if(!Number.isFinite(at))return;
    // Chromium's media clock truncates to microseconds. Land inside the selected event.
    const t=Math.max(0,Math.min(data.duration,Math.ceil(at*1e6)/1e6));
    video.currentTime=t;sync(t); if(start)play();
  }
  function toggle(){if(video.paused||video.ended){if(video.ended)seek(0);play();}else video.pause();}
  const jumpButton=(cls,text,at,label)=>button(cls,text,label+'，从 '+time(at,true)+' 播放',()=>seek(at,true));
  const role=(label,cls,symbol)=>{const n=el('span','role '+cls);n.append(icon(symbol),document.createTextNode(label));return n;};
  let mode='turns',follow=true,lastActive=null,items=[],dialogFrame=null;
  function preview(frame) {
    dialogFrame=frame;
    const dialog=get('frameDialog'), img=get('frameImage');
    img.src=frame.src;img.alt='录像 '+time(frame.at,true)+' 的眼镜取景帧';
    get('frameTime').textContent=time(frame.at,true); dialog.showModal();
  }
  function render() {
    const part=document.createDocumentFragment();
    if(mode==='turns'){
      for(const t of data.turns){
        const row=el('article','record');row.dataset.key=t.id;
        const head=el('div','row-meta');
        const tag=el('span','status');tag.hidden=true;head.append(role('用户','user-role','mic'),jumpButton('stamp',time(t.userAt,true),t.userAt,'定位用户发言'),tag);
        if(t.kind==='discarded'){tag.textContent='请求已撤销';tag.hidden=false;}
        row.append(head,jumpButton('utterance',t.user,t.userAt,'播放这次提问'));
        if(t.answer){
          const reply=el('div','reply-block'), meta=el('div','row-meta'), speech=speechById.get(t.speechId);
          const at=speech?speech.start:t.answerAt;
          meta.append(role('Satori','satori-role','sound'),jumpButton('stamp',time(at,true),at,'定位助手播报'));
          const frame=framesById.get(t.frameIds[0]);if(frame){const shot=button('frame-link',null,'查看本轮取景帧 '+time(frame.at,true),()=>preview(frame));shot.append(icon('camera'));meta.append(shot);}
          reply.append(meta,jumpButton('reply',t.answer,at,'播放这条答复'));row.append(reply);
        }
        part.append(row);
      }
    }else if(mode==='speech'){
      for(const s of data.speech){
        const row=el('article','record');row.dataset.key=s.id;const meta=el('div','row-meta');
        meta.append(role(s.turnId?'Satori 答复':'设备提示',s.turnId?'satori-role':'system-role','sound'),
          jumpButton('stamp',time(s.start,true)+'–'+time(s.end,true),s.start,'播放这段语音'),
          el('span','speech-times',(s.end-s.start).toFixed(1)+' 秒'));
        row.append(meta,jumpButton('speech-text',s.text,s.start,'播放这段语音'));part.append(row);
      }
    }else{
      const grid=el('div','frames');
      for(const f of data.frames){
        const tile=button('shot',null,'查看 '+time(f.at,true)+' 取景帧',()=>preview(f));tile.dataset.key=f.id;
        const img=el('img');img.src=f.src;img.alt='';img.loading='lazy';tile.append(img,el('span',null,time(f.at,true)));grid.append(tile);
      }part.append(grid);
    }
    records.replaceChildren(part);items=[...records.querySelectorAll('[data-key]')];lastActive=null;records.scrollTop=0;sync(video.currentTime);
  }
  function changeMode(tab){
    mode=tab.dataset.mode;tabs.forEach(t=>{const selected=t===tab;t.setAttribute('aria-selected',String(selected));t.tabIndex=selected?0:-1;});
    records.setAttribute('aria-labelledby',tab.id);render();
  }
  tabs.forEach((tab,i)=>{
    tab.addEventListener('click',()=>changeMode(tab));
    tab.addEventListener('keydown',ev=>{
      const target=ev.key==='ArrowRight'?(i+1)%tabs.length:ev.key==='ArrowLeft'?(i+tabs.length-1)%tabs.length:ev.key==='Home'?0:ev.key==='End'?tabs.length-1:null;
      if(target!==null){ev.preventDefault();tabs[target].focus();changeMode(tabs[target]);}
    });
  });
  const updateTimeline=window.createReplayTimeline(get('timeline'),data,t=>seek(t));
  function sync(t) {
    if(!Number.isFinite(t))return;
    get('timeNow').textContent=time(t);scrub.value=String(t);scrub.setAttribute('aria-valuetext',time(t,true));updateTimeline(t);
    const turn=data.turns.find(x=>x.userAt<=t&&t<x.endAt);
    const speech=data.speech.find(x=>x.start<=t&&t<x.end);
    const frame=mode==='frames'?[...data.frames].reverse().find(x=>x.at<=t&&t<x.at+2):null;
    const key=mode==='turns'?turn?.id:mode==='speech'?speech?.id:frame?.id;
    const during=mode==='turns'&&speech?.turnId&&speech.turnId===turn?.id;
    for(const row of items){
      const on=!!key&&key===row.dataset.key;
      row.classList.toggle('current',on);row.classList.toggle('speaking',!!(on&&(mode==='speech'||during)));
      if(mode==='turns'){
        const tag=row.querySelector('.status'), rec=data.turns.find(x=>x.id===row.dataset.key);
        const discarded=rec.kind==='discarded'&&t>=rec.endAt;
        tag.hidden=!on&&!discarded&&rec.kind!=='discarded';
        tag.textContent=on?(during?'播报区间':rec.answerAt!=null&&t>=rec.answerAt?'答复已生成':'处理区间'):rec.kind==='discarded'?'请求已撤销':'';
      }
      if(on&&key!==lastActive&&follow){
        const box=row.getBoundingClientRect(), panel=records.getBoundingClientRect();
        if(box.top<panel.top||box.bottom>panel.bottom)records.scrollTop+=box.top-panel.top-2;
      }
    }
    lastActive=key;
  }
  function updatePlayer(){
    const running=!video.paused&&!video.ended;
    get('playState').textContent=video.ended?'播放结束':running?'播放中':'已暂停';
    get('playState').classList.toggle('is-playing',running);
    const toggleButton=get('togglePlay');
    toggleButton.replaceChildren(icon(running?'pause':'play'));toggleButton.setAttribute('aria-label',running?'暂停视频':'播放视频');
    get('centerPlay').hidden=running||video.currentTime>0||!!video.error;
  }
  get('closeFrame').addEventListener('click',()=>get('frameDialog').close());
  get('frameSeek').addEventListener('click',()=>{get('frameDialog').close();if(dialogFrame)seek(dialogFrame.at,true);});
  get('follow').addEventListener('click',()=>{follow=!follow;get('follow').setAttribute('aria-pressed',String(follow));lastActive=null;sync(video.currentTime);});
  get('centerPlay').addEventListener('click',toggle);get('togglePlay').addEventListener('click',toggle);video.addEventListener('click',toggle);
  get('mute').addEventListener('click',()=>{video.muted=!video.muted;});
  video.addEventListener('volumechange',()=>{get('mute').replaceChildren(icon(video.muted?'muted':'volume'));get('mute').setAttribute('aria-label',video.muted?'取消静音':'静音');get('mute').title=video.muted?'取消静音':'静音';});
  get('speed').addEventListener('change',ev=>{video.playbackRate=Number(ev.target.value);});
  get('full').addEventListener('click',()=>{if(document.fullscreenElement)document.exitFullscreen();else if(get('player').requestFullscreen)get('player').requestFullscreen().catch(()=>{});else if(video.webkitEnterFullscreen)video.webkitEnterFullscreen();});
  document.addEventListener('fullscreenchange',()=>get('full').setAttribute('aria-label',document.fullscreenElement?'退出全屏':'进入全屏'));
  get('scrub').addEventListener('input',ev=>seek(Number(ev.target.value)));
  ['timeupdate','seeked','seeking','loadedmetadata'].forEach(n=>video.addEventListener(n,()=>sync(video.currentTime)));
  ['play','pause','ended','loadedmetadata'].forEach(n=>video.addEventListener(n,updatePlayer));
  video.addEventListener('error',()=>{get('mediaError').hidden=false;updatePlayer();});
  document.addEventListener('keydown',ev=>{
    if(ev.code==='Space'&&!ev.repeat&&!get('frameDialog').open&&(ev.target===document.body||ev.target===video)){ev.preventDefault();toggle();}
  });
  video.controls=false;
  get('totalTurns').textContent=data.turns.length;get('totalSpeech').textContent=data.speech.length;
  tabs.forEach(t=>t.querySelector('span').textContent=data[t.dataset.mode].length);
  render();updatePlayer();
})();
