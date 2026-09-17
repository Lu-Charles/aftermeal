const $ = id => document.getElementById(id);
let state = {sessions:[]}, sessionId = '', servingId = '', correction = null, saving = false;
const retries = new WeakMap();
const grams = value => (value / 1000).toLocaleString(undefined, {maximumFractionDigits:3});
const countLabel = (count, singular, plural=singular+'s') => `${count} ${count===1?singular:plural}`;
const fields = form => Object.fromEntries(new FormData(form));
const session = () => state.sessions.find(s => s.id === sessionId);
const serving = () => session()?.servings.find(s => s.id === servingId);
const materialNames = {portion:'Food portion',residue:'Thin residue / crumbs',empty:'Clean, empty plate',uncertain:'Uncertain'};
function element(tag, text, className) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; }
function status(message, error=false) { $('capture-status').textContent = message; $('capture-status').classList.toggle('error', error); }
function persistSelection() { history.replaceState(null,'',`#${sessionId}${servingId ? ':'+servingId : ''}`); }
async function load() {
 const response = await fetch('/api/capture');
 const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Could not load collection.');
 state = data;
 if (!session()) { sessionId = state.sessions.at(-1)?.id || ''; servingId = ''; }
 if (!serving()) servingId = '';
 render(); persistSelection();
}
function resetReading() {
 correction = null; $('reading-form').reset(); clearPreview($('reading-form'));
 $('reading-form-title').textContent = 'Add an after reading'; $('save-reading').textContent = 'Save after reading';
 $('after-photo-field').hidden = false; $('reading-form').elements.photo.required = true;
 $('reading-form').elements.note.required = false; $('cancel-correction').hidden = true; $('remaining-net').textContent = '';
}
function clearPreview(form) { const image = form.querySelector('img'); if (image) { if (image.src.startsWith('blob:')) URL.revokeObjectURL(image.src); image.removeAttribute('src'); image.hidden = true; } }
function selectSession(id) { sessionId = id; servingId = ''; resetReading(); $('serving-form').hidden = true; render(); persistSelection(); }
function selectServing(id) { servingId = id; resetReading(); $('serving-form').hidden = true; render(); persistSelection(); }
function render() {
 document.querySelector('.capture-layout').classList.toggle('empty-collection',state.sessions.length===0);
 $('new-session').hidden=state.sessions.length===0;
 const all = state.sessions.flatMap(s => s.servings.map(v => ({...v,purpose:s.purpose})));
 const active = all.filter(s => !s.exclusion);
 const complete = active.filter(s => s.purpose === 'pilot' && s.readings.some(r => r.role === 'after')).length;
 $('pilot-count').textContent = `${complete} / 10`;
 $('pair-count').textContent = `${countLabel(active.length,'active serving')} · ${countLabel(active.reduce((n,s) => n+s.readings.filter(r=>r.role==='after').length,0),'photo pair')} · ${all.length-active.length} excluded`;
 $('session-list').replaceChildren();
 for (const s of state.sessions) {
  const button = element('button',s.name,'session-item'); button.setAttribute('aria-pressed',String(s.id===sessionId));
  button.append(element('small',`${s.purpose==='pilot'?'Pilot / development':'Reserved evaluation'} · ${countLabel(s.servings.length,'serving')}`));
  button.addEventListener('click',()=>selectSession(s.id)); $('session-list').append(button);
 }
 $('session-form').hidden = state.sessions.length > 0;
 $('session-workspace').hidden = !session(); if (!session()) return;
 $('session-title').textContent = session().name;
 $('session-purpose').textContent = session().purpose === 'evaluation' ? 'Reserved evaluation · keep these servings out of training and tuning.' : 'Pilot / development · use these servings to check the collection process.';
 $('serving-list').replaceChildren();
 for (const s of session().servings) {
  const button = element('button',s.food,'serving-item'); button.setAttribute('aria-pressed',String(s.id===servingId));
  const count = s.readings.filter(r=>r.role==='after').length;
  button.append(element('small',s.exclusion ? 'Excluded · retained for history' : `${count} after reading${count===1?'':'s'}${count ? '' : ' · needs an after photo'}`));
  button.addEventListener('click',()=>selectServing(s.id)); $('serving-list').append(button);
 }
 $('empty-session').hidden = session().servings.length > 0;
 $('serving-detail').hidden = !serving(); if (!serving()) return;
 const s = serving(); $('food-title').textContent = s.food;
 $('serving-meta').textContent = `Empty plate ${grams(s.tare_mg)} g · scale increment ${grams(s.resolution_mg)} g`;
 $('excluded').hidden = !s.exclusion; $('excluded').textContent = s.exclusion ? `Excluded: ${s.exclusion}` : '';
 $('reading-form').hidden = !!s.exclusion; $('exclude-form').closest('details').hidden = !!s.exclusion;
 $('reading-list').replaceChildren();
 for (const r of s.readings) {
  const row = element('article',undefined,'reading'); const image = element('img'); image.src=r.image.url; image.alt=`${r.role==='before'?'Starting':'After'} photo of ${s.food}`;
  const content = element('div'); content.append(element('p',r.role==='before'?'Starting portion':materialNames[r.material]));
  content.append(element('strong',`${grams(r.net_mg)} g food${r.role==='after' ? ' · '+(r.remaining_fraction*100).toFixed(1)+'% remaining' : ''}`));
  content.append(element('p',`Scale display: ${grams(r.gross_mg)} g · ${new Date(r.created).toLocaleString()}`,'help'));
  if (r.below_scale_resolution) content.append(element('p','Below one scale increment; this does not establish absolutely zero food.','help'));
  if (r.note) content.append(element('p',r.note));
  if (r.role==='after' && !s.exclusion) { const button=element('button','Correct reading','text-button'); button.addEventListener('click',()=>editReading(r)); content.append(button); }
  if (r.history.length > 1) { const details=element('details'); details.append(element('summary',`${r.history.length} saved revisions`)); const list=element('ol'); for (const h of r.history) list.append(element('li',`${grams(h.gross_mg)} g · ${materialNames[h.material]} · ${h.note} · ${new Date(h.created).toLocaleString()}`)); details.append(list); content.append(details); }
  row.append(image,content); $('reading-list').append(row);
 }
}
function editReading(reading) {
 resetReading(); correction=reading;
 $('reading-form-title').textContent='Correct scale reading'; $('save-reading').textContent='Save correction';
 $('reading-form').elements.gross_g.value=reading.gross_mg/1000; $('reading-form').elements.material.value=reading.material;
 $('reading-form').elements.note.required=true; $('after-photo-field').hidden=true; $('reading-form').elements.photo.required=false;
 $('cancel-correction').hidden=false; updateRemaining(); $('reading-form').elements.gross_g.focus();
}
async function photoData(file) {
 if (!(file instanceof File) || !file.size) throw new Error('Choose a photo.');
 if (file.size > 8*1024*1024) throw new Error('Photos must be smaller than 8 MB.');
 return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=()=>reject(new Error('Could not read photo.'));reader.readAsDataURL(file);});
}
async function save(form, data, done) {
 if (saving) return;
 saving=true; const controls=[...document.querySelectorAll('button,input,select,textarea')]; controls.forEach(c=>c.disabled=true); status('Saving…');
 try {
  const payload={...data}; if ('photo' in payload) payload.photo=await photoData(payload.photo);
  const fingerprint=JSON.stringify(payload); const old=retries.get(form);
  const requestId=old?.fingerprint===fingerprint ? old.id : crypto.randomUUID(); retries.set(form,{fingerprint,id:requestId});
  const response=await fetch('/api/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...payload,request_id:requestId})});
  const result=await response.json(); if(!response.ok) throw new Error(result.error || 'Could not save this record.');
  retries.delete(form); done(result); await load(); status('Saved on this computer.');
 } catch(error) { status(error.message || 'Could not save. Your entries are still here; retry when ready.',true); }
 finally { saving=false; controls.forEach(c=>c.disabled=false); }
}
$('new-session').addEventListener('click',()=>{ $('session-form').hidden=false; $('session-form').elements.name.focus(); });
$('new-serving').addEventListener('click',()=>{ $('serving-form').hidden=false; $('serving-form').elements.food.focus(); });
$('cancel-correction').addEventListener('click',resetReading);
$('session-form').addEventListener('submit',event=>{event.preventDefault(); const form=event.currentTarget; save(form,{action:'session',...fields(form)},r=>{sessionId=r.session_id;servingId='';form.reset();});});
$('serving-form').addEventListener('submit',event=>{event.preventDefault(); const form=event.currentTarget; save(form,{action:'serving',session_id:sessionId,...fields(form)},r=>{servingId=r.serving_id;form.reset();clearPreview(form);form.hidden=true;resetReading();$('starting-net').textContent='';});});
$('reading-form').addEventListener('submit',event=>{event.preventDefault(); const form=event.currentTarget;const data=fields(form);if(correction)delete data.photo;save(form,{action:correction?'correction':'reading',serving_id:servingId,...data,...(correction?{reading_id:correction.id,expected_revision:correction.revision}:{})},resetReading);});
$('exclude-form').addEventListener('submit',event=>{event.preventDefault(); const form=event.currentTarget; save(form,{action:'exclude',serving_id:servingId,...fields(form)},()=>{form.reset();resetReading();});});
function updateRemaining() { const value=$('reading-form').elements.gross_g.value; const s=serving(); if(!s || value==='') {$('remaining-net').textContent='';return;} const net=Number(value)*1000-s.tare_mg; const start=s.readings.find(r=>r.role==='before').net_mg; $('remaining-net').textContent=net>=0&&net<=start ? `${grams(net)} g food · ${(net/start*100).toFixed(1)}% remaining from scale readings` : 'Check the scale reading against the empty plate and starting weight.'; }
$('reading-form').elements.gross_g.addEventListener('input',updateRemaining);
for (const name of ['tare_g','gross_g']) $('serving-form').elements[name].addEventListener('input',()=>{const f=$('serving-form').elements; $('starting-net').textContent=f.tare_g.value!==''&&f.gross_g.value!=='' ? `${grams((Number(f.gross_g.value)-Number(f.tare_g.value))*1000)} g starting food` : '';});
for (const input of document.querySelectorAll('input[type=file]')) input.addEventListener('change',()=>{ const img=input.parentElement.querySelector('img');if(img.src.startsWith('blob:'))URL.revokeObjectURL(img.src);const file=input.files[0];img.hidden=!file;if(file)img.src=URL.createObjectURL(file);else img.removeAttribute('src');});
[sessionId,servingId='']=location.hash.slice(1).split(':');
load().catch(error=>status(error.message,true));
