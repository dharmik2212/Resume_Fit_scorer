const API_BASE = '';
let uploadedFiles = [];
let analysisSessionId = null;

const $ = id => document.getElementById(id);
const jdText = $('jdText'), charCount = $('charCount'), submitBtn = $('submitBtn');
const dropZone = $('dropZone'), fileInput = $('resumeFiles'), fileList = $('fileList');
const emptyState = $('emptyState'), loadingState = $('loadingState'), resultsContent = $('resultsContent');
const resultsList = $('resultsList'), resultsMeta = $('resultsMeta'), resultsStats = $('resultsStats');
const chatForm = $('chatForm'), chatInput = $('chatInput'), chatMessages = $('chatMessages');

function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}

function toast(msg){
  document.querySelector('.toast')?.remove();
  const t=document.createElement('div');t.className='toast';t.textContent=msg;
  document.body.appendChild(t);setTimeout(()=>t.remove(),4000);
}

function fmtBytes(b){
  if(b<1024) return b+' B';
  if(b<1048576) return (b/1024).toFixed(1)+' KB';
  return (b/1048576).toFixed(1)+' MB';
}

function renderFiles(){
  fileList.innerHTML='';
  uploadedFiles.forEach((f,i)=>{
    const c=document.createElement('div');c.className='file-chip';
    c.innerHTML=`<span class="file-chip-name">${esc(f.name)}</span><span class="file-chip-size">${fmtBytes(f.size)}</span><button class="file-chip-remove" type="button">×</button>`;
    c.querySelector('button').onclick=e=>{e.stopPropagation();uploadedFiles.splice(i,1);renderFiles()};
    fileList.appendChild(c);
  });
}

function addFiles(files){
  const allowed=['pdf','docx','txt'];
  for(const f of files){
    const ext=f.name.split('.').pop().toLowerCase();
    if(!allowed.includes(ext)){toast(f.name+' is not supported. Use PDF, DOCX, or TXT.');continue}
    if(f.size>10*1024*1024){toast(f.name+' exceeds 10 MB.');continue}
    if(uploadedFiles.length>=50){toast('Maximum 50 resumes.');break}
    if(!uploadedFiles.some(e=>e.name===f.name&&e.size===f.size)) uploadedFiles.push(f);
  }
  renderFiles();
}

fileInput.addEventListener('change',e=>{addFiles([...e.target.files]);e.target.value=''});
['dragenter','dragover'].forEach(n=>dropZone.addEventListener(n,e=>{e.preventDefault();dropZone.classList.add('drag-over')}));
['dragleave','drop'].forEach(n=>dropZone.addEventListener(n,e=>{e.preventDefault();dropZone.classList.remove('drag-over')}));
dropZone.addEventListener('drop',e=>addFiles([...e.dataTransfer.files]));

jdText.addEventListener('input',()=>{charCount.textContent=jdText.value.length.toLocaleString()+' characters'});
charCount.textContent='0 characters';

function animateStages(){
  const s=[$('s1'),$('s2'),$('s3')];
  s.forEach(x=>x.classList.remove('active','done'));
  s[0].classList.add('active');
  s.slice(1).forEach((x,i)=>setTimeout(()=>{s[i].classList.remove('active');s[i].classList.add('done');x.classList.add('active')},(i+1)*800));
}

function scoreCircle(score){
  const safe=Math.min(Math.max(Number(score)||0,0),100),r=24,c=2*Math.PI*r;
  const lvl=safe>=70?'high':safe>=45?'mid':'low';
  return `<div class="score-circle"><svg viewBox="0 0 58 58"><circle class="score-bg" cx="29" cy="29" r="24"/><circle class="score-fill s-${lvl}" cx="29" cy="29" r="24" stroke-dasharray="${c}" stroke-dashoffset="${c-safe/100*c}"/></svg><div class="score-text"><span class="score-${lvl}">${Math.round(safe)}</span><span class="score-label">score</span></div></div>`;
}

function skillChips(items,cls,empty){
  if(!items?.length) return `<span class="s-empty">${empty}</span>`;
  return items.map(s=>`<span class="skill-chip ${cls}">${esc(s)}</span>`).join('');
}

function renderResults(data){
  analysisSessionId=data.session_id;
  const s=data.stage_summary;
  resultsStats.innerHTML=[
    ['Input',s.input],['Parsed',s.parsed],['Keyword Pass',s.after_stage1_bm25],['Deep Scored',s.after_stage3_llm]
  ].map(([l,v])=>`<div class="r-stat"><span class="r-stat-val">${v}</span><span class="r-stat-label">${l}</span></div>`).join('');

  const cost=data.cost_estimate_usd>0?'~$'+data.cost_estimate_usd.toFixed(5)+' API cost':'$0 external cost';
  resultsMeta.textContent=data.processed_time_seconds+'s · '+cost+' · '+data.results.length+' candidates';

  resultsList.innerHTML='';
  data.results.forEach((r,i)=>{
    const sem=r.semantic_score==null?null:Math.round(r.semantic_score*100);
    const card=document.createElement('div');card.className='result-card'+(i<3?' rank-'+(i+1):'');
    const sl=r.stage_reached===3?'Deep scored':r.stage_reached===2?'Semantic stage':'Keyword stage';
    const skillPct = r.breakdown ? Math.round(r.breakdown.skills_match) : (r.bm25_score ? Math.round(r.bm25_score) : '?');
    const eduDetail = r.education_detail ? esc(r.education_detail).substring(0,90) : (r.education_label ? esc(r.education_label) : 'N/A');
    const educationDisplay = r.education_detail ? esc(r.education_detail).replace(/\s*\|\s*/g, ' | ') : (r.education_label || 'N/A');
    const companiesDisplay = r.companies?.length ? esc(r.companies.slice(0,4).join(', ')) + (r.company_count > 4 ? ' (+'+(r.company_count-4)+' more)' : '') : 'Not parsed';
    const bd=r.breakdown?`<div class="breakdown-grid">${[['Skills Match',r.breakdown.skills_match],['Experience',r.breakdown.experience],['Education',r.breakdown.education]].map(([n,v])=>`<div class="bd-item"><div class="bd-label">${n}</div><div class="bd-val">${Math.round(v)}</div></div>`).join('')}</div>`:'';
    card.innerHTML=`<div class="result-top"><div class="rank-badge">#${r.rank}</div><div class="result-info"><div class="result-name">${esc(r.candidate_name)}</div><div class="result-filename">${esc(r.filename)}</div><span class="stage-tag stage-tag-${r.stage_reached}">${sl}</span></div>${scoreCircle(r.final_score)}</div>
      <div style="display:flex;gap:12px;padding:6px 0 0 52px;font-size:12px;color:var(--text-secondary)"><span><strong>Skills:</strong> ${skillPct}%</span></div>
      <div class="result-details" id="rd${i}">
        <div class="stage-bars">${[['Keyword',Math.round(r.bm25_score),'bm25'],['Semantic',sem,'semantic'],['Deep',r.llm_score==null?null:Math.round(r.llm_score),'deep']].map(([n,v,t])=>`<div class="sbar-item"><div class="sbar-label"><span>${n}</span><span>${v??'—'}</span></div><div class="sbar-bar"><div class="sbar-fill sbar-${t}" style="width:${v??0}%"></div></div></div>`).join('')}</div>
        <div class="skills-grid"><div><div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#047857;margin-bottom:4px">Matched Skills</div><div class="skill-chips">${skillChips(r.matched_skills,'s-matched','No explicit matches')}</div></div><div><div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;color:#92400e;margin-bottom:4px">Missing Skills</div><div class="skill-chips">${skillChips(r.missing_skills,'s-missing','No named gaps')}</div></div></div>
        <div class="reason-text"><strong>Details:</strong> ${esc(r.reason)}</div>${bd}
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);font-size:13px;color:var(--text-secondary);display:flex;flex-direction:column;gap:4px">
          <div><strong>Education:</strong> ${educationDisplay}</div>
          <div><strong>Companies:</strong> ${companiesDisplay}</div>
        </div>
      </div>`;
    card.querySelector('.result-top').addEventListener('click',()=>{$('rd'+i).classList.toggle('open')});
    resultsList.appendChild(card);
  });

  loadingState.classList.add('hidden');
  resultsContent.classList.remove('hidden');
  if(data.warnings?.length) toast(data.warnings.join(' '));
}

submitBtn.addEventListener('click',async ()=>{
  const jd=jdText.value.trim();
  if(jd.length<40) return toast('Job description needs at least 40 characters.');
  if(!uploadedFiles.length) return toast('Upload at least one resume.');

  emptyState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  loadingState.classList.remove('hidden');
  submitBtn.disabled=true;
  animateStages();

  const fd=new FormData();
  fd.append('jd_text',jd);
  uploadedFiles.forEach(f=>fd.append('resumes',f));

  try{
    const r=await fetch(API_BASE+'/api/score',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Analysis failed.');
    renderResults(d);
  }catch(e){
    toast('Error: '+e.message);
    loadingState.classList.add('hidden');
    emptyState.classList.remove('hidden');
  }finally{submitBtn.disabled=false}
});

function addMsg(text,role){
  const m=document.createElement('div');m.className='msg msg-'+role;m.textContent=text;
  chatMessages.appendChild(m);chatMessages.scrollTop=chatMessages.scrollHeight;
}

async function ask(q){
  if(!analysisSessionId||!q.trim()) return;
  addMsg(q.trim(),'user');chatInput.value='';chatForm.querySelector('.btn').disabled=true;
  try{
    const r=await fetch(API_BASE+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:analysisSessionId,question:q.trim()})});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Could not answer.');
    addMsg(d.answer,'assistant');
  }catch(e){addMsg(e.message,'error')}
  finally{chatForm.querySelector('.btn').disabled=false;chatInput.focus()}
}

chatForm.addEventListener('submit',e=>{e.preventDefault();ask(chatInput.value)});
document.querySelectorAll('[data-q]').forEach(b=>b.addEventListener('click',()=>ask(b.dataset.q)));
