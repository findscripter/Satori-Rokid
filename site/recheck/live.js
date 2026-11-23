(() => {
  'use strict';
  const get=id=>document.getElementById(id);
  const video=get('liveVideo');
  const seq=get('liveSeq'), model=get('liveModel'), mode=get('liveMode');
  const list=get('liveList'), toast=get('liveToast'), next=get('liveNext');
  const rows=window.STUDY.rows;
  const lifetime=7, preroll=2.2, epsilon=0.000001;
  const phaseNames={direct:'直接交付',candidate:'候选 · 待复核',keep:'复核保留 · 交付',hold:'暂缓 · 未交付',cancel:'取消 · 未交付'};
  let events=[],eventButtons=[],sourceSeq=null,pendingTime=null;
  let painted=null,seen=-1,shown=[],dismissed=new Set(),frameCallback=null;
  const node=(tag,cls,text)=>{const n=document.createElement(tag);n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const clock=(value,fine=false)=>{
    const tenths=Math.floor((Math.max(0,value)+epsilon)*10);
    return String(Math.floor(tenths/600)).padStart(2,'0')+':'+String(Math.floor(tenths/10)%60).padStart(2,'0')+(fine?'.'+tenths%10:'');
  };
  function schedule(){
    const result=[];
    for(const r of rows){
      if(r.seq!==Number(seq.value)||(model.value!=='both'&&r.model!==model.value))continue;
      const add=(phase,at,until=at+lifetime)=>result.push({key:r.id+'@'+phase,r,phase,at,until});
      if(mode.value==='direct')add('direct',r.d0);
      else if(mode.value==='gated'){if(r.verdict==='keep')add('keep',r.d1);}
      else{
        add('candidate',r.d0,r.d1);
        add(r.verdict,r.d1);
      }
    }
    return result.sort((a,b)=>a.at-b.at||a.key.localeCompare(b.key));
  }
  const caption=e=>e.phase==='hold'||e.phase==='cancel'?e.r.reasonZh:e.r.messageZh;
  function renderList(){
    events=schedule();dismissed.clear();painted=null;seen=-1;
    const fragment=document.createDocumentFragment();
    eventButtons=events.map(e=>{
      const b=node('button','live-event');b.type='button';
      b.dataset.at=String(e.at);b.dataset.phase=e.phase;b.dataset.rowId=e.r.id;
      const head=node('span','live-event-head');
      head.append(node('time','live-event-time',clock(e.at,true)),node('span','live-event-model',e.r.model),node('span','live-phase '+e.phase,phaseNames[e.phase]));
      b.append(head,node('span','live-event-text',caption(e)));
      b.setAttribute('aria-label',clock(e.at,true)+' '+e.r.model+' '+phaseNames[e.phase]+'。'+caption(e));
      b.onclick=()=>playFrom(e.at);
      fragment.append(b);return b;
    });
    if(!events.length)fragment.append(node('p','live-empty','这个组合没有对应的交互。'));
    list.replaceChildren(fragment);sync(pendingTime??video.currentTime);
  }
  function sync(time){
    if(!Number.isFinite(time))return;
    const now=time+epsilon;
    const n=events.findIndex(e=>e.at>now),passed=n<0?events.length:n;
    const active=events.filter(e=>e.at<=now&&now<e.until);
    const keys=new Set(active.map(e=>e.key));
    for(const key of dismissed)if(!keys.has(key))dismissed.delete(key);
    shown=active.filter(e=>!dismissed.has(e.key));
    const label=clock(time,true)+(Number.isFinite(video.duration)?' / '+clock(video.duration):'');
    if(get('liveTime').textContent!==label)get('liveTime').textContent=label;
    if(passed!==seen){
      seen=passed;get('liveCount').textContent='已到达 '+passed+' / '+events.length+' 个事件';
      next.disabled=passed===events.length;
      next.textContent=next.disabled?'已到最后一次交互':'播放下一次交互 · '+clock(events[passed].at,true);
    }
    const signature=active.map(e=>e.key).join('|')+' / '+shown.map(e=>e.key).join('|');
    if(signature===painted)return;
    painted=signature;
    const visibleKeys=new Set(shown.map(e=>e.key));
    for(let i=0;i<events.length;i++){
      const current=keys.has(events[i].key);
      eventButtons[i].classList.toggle('is-current',current);
      if(current)eventButtons[i].setAttribute('aria-current','time'); else eventButtons[i].removeAttribute('aria-current');
    }
    if(active.length){
      const latest=eventButtons[events.indexOf(active[active.length-1])],box=list.getBoundingClientRect(),item=latest.getBoundingClientRect();
      if(item.top<box.top||item.bottom>box.bottom)list.scrollTop+=item.top-box.top-2;
    }
    toast.hidden=visibleKeys.size===0;
    if(!shown.length)return;
    get('toastMeta').textContent=mode.value==='audit'?'复核过程':'助手提醒';
    get('toastTime').textContent=clock(shown[shown.length-1].at,true);
    const fragment=document.createDocumentFragment();
    for(const e of shown){
      const msg=node('div','live-bubble '+e.phase),head=node('div','live-bubble-head');
      head.append(node('b','',e.r.model),node('span','live-phase '+e.phase,phaseNames[e.phase]),node('time','live-bubble-at',clock(e.at,true)));
      msg.append(head,node('p','live-bubble-text',caption(e)));fragment.append(msg);
    }
    get('toastText').replaceChildren(fragment);
  }
  function notice(text){get('liveNotice').textContent=text;}
  const release='https://github.com/findscripter/Satori-Rokid/releases/download/recheck-4k-2026-11-23/';
  const localVideos=new Map();
  function setSource(){
    if(sourceSeq===seq.value)return;
    sourceSeq=seq.value;pendingTime=0;dismissed.clear();painted=null;
    video.pause();
    const stem='sequence-'+String(sourceSeq).padStart(2,'0');
    video.poster='media/'+stem+'-poster.jpg';
    video.src=localVideos.get(sourceSeq)||((location.protocol==='file:'?'media/':release)+stem+'-4k.mp4');
    video.preload='metadata';video.load();
    notice('正在读取 4K 视频…');
  }
  function play(){
    const task=video.play();
    if(task)task.catch(e=>{if(e.name!=='AbortError'&&!video.error)notice('请点击视频播放键继续。');});
  }
  function seek(at){
    const clipped=Math.max(0,Math.ceil(at*1e6)/1e6);
    dismissed.clear();painted=null;
    if(video.readyState>=1){pendingTime=null;video.currentTime=Math.min(clipped,video.duration||Infinity);}
    else pendingTime=clipped;
    sync(clipped);
  }
  function playFrom(at){seek(at);play();}
  next.onclick=()=>{
    const t=pendingTime??video.currentTime;
    const e=events.find(e=>e.at>t+epsilon);
    if(e)playFrom(Math.max(t,e.at-preroll));
  };
  get('dismissToast').onclick=()=>{
    for(const e of shown)dismissed.add(e.key);
    sync(video.currentTime);
  };
  seq.addEventListener('change',()=>{setSource();renderList();});
  get('localVideoFile').addEventListener('change',e=>{
    const file=e.target.files[0];if(!file)return;
    const previous=localVideos.get(seq.value);
    localVideos.set(seq.value,URL.createObjectURL(file));sourceSeq=null;
    setSource();renderList();if(previous)URL.revokeObjectURL(previous);
    e.target.value='';
  });
  model.addEventListener('change',renderList);
  mode.addEventListener('change',renderList);
  get('liveSpeed').addEventListener('change',e=>{video.playbackRate=Number(e.target.value);});
  get('liveFull').onclick=async()=>{
    try{
      if(document.fullscreenElement)await document.exitFullscreen();
      else if(get('liveStage').requestFullscreen)await get('liveStage').requestFullscreen();
      else notice('当前浏览器未提供带交互层的全屏播放。');
    }catch{notice('全屏未能开启，可继续在页面中播放。');}
  };
  document.addEventListener('fullscreenchange',()=>{
    const full=document.fullscreenElement===get('liveStage');
    get('liveFull').textContent=full?'退出全屏 ⛶':'全屏 ⛶';
    get('liveFull').setAttribute('aria-label',full?'退出全屏':'进入全屏');get('liveFull').setAttribute('aria-pressed',String(full));
  });
  video.addEventListener('loadedmetadata',()=>{
    const expected=seq.value==='1'?1672.537533:2468.399267;
    if(Math.abs(video.duration-expected)>0.2||video.videoWidth!==3840||video.videoHeight!==2160){
      video.pause();video.removeAttribute('src');video.load();
      const previous=localVideos.get(seq.value);
      if(previous){localVideos.delete(seq.value);URL.revokeObjectURL(previous);}
      notice('文件的时长或分辨率不匹配，请选择当前录像对应的完整 4K 文件。');return;
    }
    const at=pendingTime;pendingTime=null;
    if(at!==null&&at>0)seek(at);
    video.playbackRate=Number(get('liveSpeed').value);
    notice(video.videoWidth+' × '+video.videoHeight+' · 无声第一视角录像');
    sync(video.currentTime);
  });
  video.addEventListener('error',()=>{
    toast.hidden=true;notice('视频加载失败。可从页面下方下载当前录像，再选择本地 4K 文件播放。');
  });
  video.addEventListener('seeking',()=>{painted=null;dismissed.clear();sync(pendingTime??video.currentTime);});
  for(const type of ['timeupdate','seeked','pause','ratechange','ended'])video.addEventListener(type,()=>sync(pendingTime??video.currentTime));
  function nextFrame(){
    frameCallback=null;
    sync(pendingTime??video.currentTime);
    if(!video.paused&&!video.ended)frameCallback=video.requestVideoFrameCallback(nextFrame);
  }
  video.addEventListener('playing',()=>{
    if(video.requestVideoFrameCallback&&frameCallback===null)frameCallback=video.requestVideoFrameCallback(nextFrame);
  });
  // Fullscreen the stage so its synchronized interaction layer remains visible.
  video.controlsList?.add('nofullscreen');
  video.addEventListener('dblclick',event=>{
    event.preventDefault();
    get('liveFull').onclick();
  });
  window.openLiveCase=r=>{
    if(!r||!rows.some(x=>x.id===r.id))return;
    seq.value=String(r.seq);model.value=r.model;mode.value='audit';
    setSource();renderList();get('liveStage').scrollIntoView({block:'center'});playFrom(Math.max(0,r.d0-preroll));
  };
  setSource();renderList();
})();
