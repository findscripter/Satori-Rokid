(() => {
  'use strict';
  const {rows,summary}=window.STUDY;
  const el=id=>document.getElementById(id);
  const N=(tag,cls='',text=null)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==null)e.textContent=text;return e;};
  const filters=['model','seq','verdict','special'];
  const labels={valid:'仍适用',obsolete:'已过时',unknown:'无法确定',withheld:'未交付'};
  const verdicts={keep:'保留',hold:'暂缓',cancel:'取消'};
  const nameSeq=n=>'录像'+(Number(n)===1?'一':'二');
  const clock=(num,digits=0)=>{
    const scale=10**digits,total=Math.floor(num*scale+1e-7),sec=Math.floor(total/scale);
    return String(Math.floor(sec/60)).padStart(2,'0')+':'+String(sec%60).padStart(2,'0')+(digits?'.'+String(total%scale).padStart(digits,'0'):'');
  };
  const rowDefault=rows.find(x=>x.model==='Sol'&&x.seq===1&&x.t===180)||rows[0];
  let current=rowDefault,asPct=false;
  function drawChart(){
    el('asPct').setAttribute('aria-pressed',String(asPct));el('asCount').setAttribute('aria-pressed',String(!asPct));
    const frag=document.createDocumentFragment();
    for(const [title,values] of [['直接交付',summary.direct],['等时延交付',summary.delay],['复核后交付',summary.gate]]){
      const n=values.valid+values.obsolete+values.unknown;
      const row=N('div','chart-row'), track=N('div','track'),all={...values,withheld:rows.length-n};
      track.setAttribute('role','img');track.setAttribute('aria-label',title+'，'+Object.entries(all).map(([k,v])=>labels[k]+' '+v+' 条').join('，'));
      for(const [key,count] of Object.entries(all)){
        if(!count)continue;
        const rate=100*count/rows.length;
        const piece=N('span','piece '+key,asPct?rate.toFixed(1)+'%':String(count));
        piece.style.width=rate+'%';piece.title=title+' · '+labels[key]+' '+count+' 条，占全部候选 '+rate.toFixed(1)+'%';track.append(piece);
      }
      row.append(N('b','',title),track,N('span','','交付 '+n));frag.append(row);
    }
    el('chart').replaceChildren(frag);
  }
  el('asPct').onclick=()=>{asPct=true;drawChart();};el('asCount').onclick=()=>{asPct=false;drawChart();};
  function matches(r){
    const special=el('special').value;
    return (!el('model').value||el('model').value===r.model)&&(!el('seq').value||Number(el('seq').value)===r.seq)&&
      (!el('verdict').value||el('verdict').value===r.verdict)&&
      (!special||(special==='intercept'&&r.label0==='obsolete'&&r.verdict!=='keep')||
       (special==='late'&&r.label0==='valid'&&r.verdict==='keep'&&r.label1==='obsolete')||
       (special==='miss'&&r.label0==='valid'&&r.verdict==='hold'));
  }
  function renderList(reveal=false){
    const visible=rows.filter(matches).sort((a,b)=>a.seq-b.seq||a.t-b.t||a.model.localeCompare(b.model));
    el('caseCount').textContent=visible.length+' / '+rows.length+' 条';
    el('empty').hidden=!!visible.length;el('detail').hidden=!visible.length;
    if(!visible.length){el('list').replaceChildren();current=null;return;}
    if(!current||!visible.some(r=>r.id===current.id))current=visible[0];
    const frag=document.createDocumentFragment();
    for(const r of visible){
      const b=N('button','item');b.type='button';b.dataset.id=r.id;
      const top=N('span','item-top');top.append(N('time','',clock(r.t)),N('span','',nameSeq(r.seq)+' · '+r.model),N('span','mini '+r.verdict,verdicts[r.verdict]));
      b.append(top,N('span','text',r.messageZh));b.title=r.messageZh;
      b.onclick=()=>{current=r;markSelection();renderDetails();};
      frag.append(b);
    }
    el('list').replaceChildren(frag);markSelection();renderDetails();
    if(reveal){
      const active=el('list').querySelector('.on'),area=el('list');
      const delta=active.getBoundingClientRect().top-area.getBoundingClientRect().top;
      if(delta<0||delta+active.offsetHeight>area.clientHeight)area.scrollTop+=delta;
    }
  }
  function markSelection(){for(const b of el('list').querySelectorAll('button')){const on=b.dataset.id===current?.id;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));}}
  function showImage(r,idx){
    const title=(idx?'复核完成':'原可交付')+' · '+clock(idx?r.d1:r.d0,2)+' · '+nameSeq(r.seq);
    el('lightTitle').textContent=title;el('lightImg').alt=title+'的第一视角图像';el('lightImg').src=idx?r.thumb1:r.thumb0;
    el('light').showModal();
  }
  el('closeLight').onclick=()=>el('light').close();
  el('light').addEventListener('click',e=>{if(e.target===el('light')){const b=el('light').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)el('light').close();}});
  function drawFrames(r){
    const frag=document.createDocumentFragment();
    for(const idx of [0,1]){
      const title=idx?'复核完成时':'原可交付时',t=idx?r.d1:r.d0, state=idx?r.label1:r.label0;
      const fig=N('figure'),head=N('div','figure-head');head.append(N('span','',title),N('time','',clock(t,2)));
      const button=N('button','image-open');button.type='button';button.setAttribute('aria-label','放大'+title+'画面 '+clock(t,2));
      const img=N('img');img.width=448;img.height=252;img.src=idx?r.thumb1:r.thumb0;img.alt=title+'的第一视角画面';button.append(img);button.onclick=()=>showImage(r,idx);
      const caption=N('figcaption');caption.append(N('span','', '盲评'),N('span','badge '+state,labels[state]));
      fig.append(head,button,caption);frag.append(fig);
    }
    el('comparison').replaceChildren(frag);
  }
  function renderDetails(){
    const r=current;el('original').open=false;el('evidence').open=false;
    el('which').textContent=nameSeq(r.seq)+'　/　GPT‑6.2 '+r.model;el('decisionPill').className='pill '+r.verdict;el('decisionPill').textContent='复核决定 · '+verdicts[r.verdict];
    el('message').textContent=r.messageZh;el('messageEn').textContent=r.message;el('created').textContent='初次判断 '+clock(r.t,2);
    el('latency').textContent='额外等待 +'+r.latency.toFixed(2)+' 秒';el('reason').textContent=r.reasonZh;
    drawFrames(r);
    const proof=document.createDocumentFragment();for(const [label,body] of [['原可交付时',r.evidence0],['复核完成时',r.evidence1],['观察到的有效时间窗',r.interval+'。'+r.intervalEvidence]]){
      const p=N('p');p.append(N('b','',label+'　'),document.createTextNode(body));proof.append(p);
    }
    el('rationale').replaceChildren(proof);
  }
  el('clipTab').onclick=()=>{if(current)window.openLiveCase(current);};
  function reset(){filters.forEach(id=>el(id).value='');current=rowDefault;renderList(true);}
  filters.forEach(id=>el(id).addEventListener('change',()=>renderList(true)));
  el('reset').onclick=reset;el('emptyReset').onclick=reset;
  drawChart();renderList(true);
})();
