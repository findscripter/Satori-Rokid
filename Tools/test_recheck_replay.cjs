// Exercise the real playback scheduler against a deterministic media clock; no network or video decoder.
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
let document;
class Node extends EventTarget {
  constructor(tag){super();this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.hidden=false;this.textContent='';this.scrollTop=0;this.value='';this.classList={toggle:(k,on)=>this.attrs[k]=on};}
  setAttribute(k,v){this.attrs[k]=v;}
  removeAttribute(k){delete this.attrs[k];}
  append(...children){for(const c of children)c.tag==='fragment'?this.children.push(...c.children):this.children.push(c);}
  replaceChildren(...children){this.children=[];this.append(...children);}
  emit(name){this.dispatchEvent(new Event(name));}
  getBoundingClientRect(){return {top:0,bottom:100};}
  scrollIntoView(){}
}
class Video extends Node{
  constructor(){super('video');this.currentTime=0;this.duration=1672.537533;this.readyState=1;this.videoWidth=3840;this.videoHeight=2160;this.paused=true;}
  load(){this.readyState=0;this.currentTime=0;}
  loaded(){this.readyState=1;this.emit('loadedmetadata');}
  play(){this.paused=false;this.emit('playing');return Promise.resolve();}
  pause(){this.paused=true;this.emit('pause');}
  tick(t){this.currentTime=t;this.emit('timeupdate');}
}
const ids={liveVideo:new Video()};
document=new Node('document');
document.createElement=t=>new Node(t);document.createDocumentFragment=()=>new Node('fragment');
document.getElementById=id=>ids[id]??=new Node(id);
const env=vm.createContext({window:{},document,console,location:{protocol:'https:'},URL});
function load(name){vm.runInContext(fs.readFileSync(path.join(__dirname,'..','site','recheck',name),'utf8'),env);}
const e=document.getElementById;
e('liveSeq').value='1';e('liveMode').value='gated';e('liveModel').value='both';e('liveSpeed').value='1';
load('data.js');load('live.js');const v=ids.liveVideo;v.loaded();
const t=env.window.STUDY.rows.find(r=>r.seq===1&&r.t===180&&r.model==='Sol');
const a=env.window.STUDY.rows.find(r=>r.seq===1&&r.t===180&&r.model==='Astra');
assert.match(v.src,/sequence-01-4k/);assert.match(v.poster,/01-poster/);
assert.equal(e('liveList').children.length,env.window.STUDY.rows.filter(r=>r.seq===1&&r.verdict==='keep').length);assert.equal(e('liveToast').hidden,true);
v.tick(t.d0);assert.equal(e('liveToast').hidden,true);
v.tick(t.d1);assert.equal(e('liveToast').hidden,false);assert.equal(e('toastText').children.length,1);
v.tick(a.d1);assert.equal(e('toastText').children.length,2);
e('dismissToast').onclick();assert.equal(e('liveToast').hidden,true);
v.tick(a.d1+.1);assert.equal(e('liveToast').hidden,true);
v.tick(0);e('liveNext').onclick();assert.equal(v.paused,false);
const next=env.window.STUDY.rows.find(r=>r.seq===1&&r.t===60);
assert.ok(Math.abs(v.currentTime-(next.d1-2.2))<.000002);v.tick(next.d1+.01);assert.equal(e('liveToast').hidden,false);
v.pause();v.emit('pause');assert.equal(e('liveToast').hidden,false);v.tick(next.d1+7.1);assert.equal(e('liveToast').hidden,true);
const cancelled=env.window.STUDY.rows.find(r=>r.model==='Sol'&&r.seq===1&&r.t===420);
env.window.openLiveCase(cancelled);assert.equal(e('liveMode').value,'audit');assert.equal(e('liveModel').value,'Sol');
v.tick(cancelled.d0);assert.equal(e('toastText').children[0].children[1].textContent,cancelled.messageZh);
v.tick(cancelled.d1);assert.equal(e('toastText').children.length,1);assert.equal(e('toastText').children[0].children[1].textContent,cancelled.reasonZh);
e('liveMode').value='gated';e('liveMode').emit('change');assert.equal(e('liveToast').hidden,true);
e('liveMode').value='direct';e('liveMode').emit('change');v.tick(cancelled.d0);assert.equal(e('liveToast').hidden,false);
e('liveSeq').value='2';e('liveSeq').emit('change');assert.match(v.src,/sequence-02-4k/);v.duration=2468.399267;v.loaded();assert.equal(e('liveToast').hidden,true);
v.tick(2490);assert.equal(e('liveNext').disabled,true);v.tick(60);assert.equal(e('liveNext').disabled,false);
const event=e('liveList').children[0];event.onclick();assert.ok(Math.abs(v.currentTime-Number(event.dataset.at))<.000002);assert.equal(e('liveToast').hidden,false);
assert.equal(v.src,'https://github.com/findscripter/Satori-Rokid/releases/download/recheck-4k-2026-11-23/sequence-02-4k.mp4');
v.duration=90;v.loaded();assert.match(e('liveNotice').textContent,/文件的时长或分辨率不匹配/);
console.log('Passed: strict event timing, overlapping models, dismiss/seek/replay, pre-roll, model and sequence filters, three display modes.');
