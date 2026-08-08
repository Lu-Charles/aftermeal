'use strict';
const $ = id => document.getElementById(id);
const codes = {F:'Visible food-like material',R:'No clear portion; residue or crumbs may remain',U:'Uncertain material'};
const flags = {
 uncertain_material:'Material is ambiguous in the after photo.',
 no_clear_portion_despite_recorded_fraction_above_10_percent:'No clear portion was identified, but the recorded remaining mass exceeds 10%.',
 visible_material_with_recorded_zero_mass:'Visible material was identified despite a recorded after weight of zero.',
 starting_photo_unclear_despite_positive_mass:'The starting photo looks unclear or empty despite a positive starting weight.',
 reviewer_note:'A separate visual-review note is available.'
};
let data,filtered=[],currentId,reviews={},storageKey,sourceViewed=false,overlayViewed=false,proposalViewed=false,exportUrl;
const pct = value => `${(value*100).toFixed(2)}%`;
function message(id,text,error=false){$(id).textContent=text;$(id).classList.toggle('error',error);}
function paragraph(text){const p=document.createElement('p');p.textContent=text;return p;}
function line(label,value){const d=document.createElement('div');d.className='evidence-line';const a=document.createElement('span'),b=document.createElement('strong');a.textContent=label;b.textContent=value;d.append(a,b);return d;}
function progress(){$('progress').textContent=`${Object.keys(reviews).length} / ${data.rows.length} reviewed`;$('export').disabled=!Object.keys(reviews).length;}
function filter(preferred){
 const q=$('search').value.trim().toLowerCase(),mode=$('filter').value;
 filtered=data.rows.filter(r=>{
  const match=mode==='all'||(mode==='flagged'&&r.flags.length)||(mode==='conflict'&&r.flags.some(f=>!['uncertain_material','reviewer_note'].includes(f)))||(mode==='pending'&&!reviews[r.record_id])||(mode==='reviewed'&&reviews[r.record_id]);
  return match&&`${r.record_id} ${r.group}`.toLowerCase().includes(q);
 });
 currentId=filtered.some(r=>r.record_id===preferred)?preferred:filtered[0]?.record_id;
 render();progress();
}
function images(){
 if(!currentId)return;message('image-status','');
 overlayViewed=overlayViewed||$('show-overlay').checked;
 for(const kind of ['before','after']){
  const base=`/api/lab/images/${currentId}/${kind}`;
  $(kind+'-photo').src=`${base}${$('show-overlay').checked?'-overlay':''}.jpg`;
  $(kind+'-full').href=`${base}.jpg`;
  $(kind+'-photo').alt=`${currentId} ${kind} eating${$('show-overlay').checked?', with predicted food highlighted in blue':''}`;
 }
}
function render(){
 const i=filtered.findIndex(r=>r.record_id===currentId),r=filtered[i];
 $('review').hidden=!r;$('empty').hidden=!!r;if(!r)return;
 $('pair-title').textContent=`${r.record_id} · ${r.group.replace(/^L:/,'')}`;
 $('pair-position').textContent=`Pair ${i+1} of ${filtered.length}${reviews[currentId]?' · Reviewed':''}`;
 $('previous').disabled=i===0;$('next').disabled=i===filtered.length-1;
 $('source-details').open=false;$('ai-details').open=false;sourceViewed=false;overlayViewed=false;proposalViewed=false;
 const saved=reviews[currentId];$('decision').value=saved?.observation||'';$('pair-issue').checked=!!saved?.pair_issue;$('review-note').value=saved?.note||'';
 $('before-decision').value=saved?.before_observation||'';
 $('remove-review').hidden=!saved;
 message('save-status',saved?'Saved in this browser. You can update this review.':'');
 const source=$('source-evidence');source.replaceChildren(line('Recorded before',`${r.recorded_before_g} g`),line('Recorded after',`${r.recorded_after_g} g`),line('Recorded remaining',pct(r.recorded_fraction)));
 for(const [name,value] of Object.entries(r.predictions))source.append(line(name==='Change'?'Image-change baseline':name,pct(value)));
 source.append(paragraph('Held-out fold predictions on development data. Recorded weights may conflict with the photographs.'));
 const ai=$('ai-evidence');ai.replaceChildren(paragraph(`Before: ${codes[r.before_visual_proposal]}`),paragraph(`After: ${codes[r.after_visual_proposal]}`));
 for(const f of r.flags)ai.append(paragraph(flags[f]||f));if(r.note)ai.append(paragraph(r.note));
 ai.append(paragraph('AI proposal only; not human-verified. No corrected weight was inferred.'));
 images();history.replaceState(null,'',`#${currentId}`);
}
async function start(){
 try{
  const response=await fetch('/api/lab');if(!response.ok)throw new Error((await response.json()).error);data=await response.json();
  storageKey=`food-vision-review-v1-${data.review_version}`;
  try{reviews=JSON.parse(localStorage.getItem(storageKey)||'{}');if(!reviews||Array.isArray(reviews)||typeof reviews!=='object')reviews={};}
  catch{reviews={};message('load-status','Browser storage is unavailable. Export reviews before closing this page.',true);}
  const ids=new Set(data.rows.map(r=>r.record_id));
  reviews=Object.fromEntries(Object.entries(reviews).filter(([id,v])=>ids.has(id)&&v&&typeof v.observation==='string'));
  const table=document.createElement('table'),caption=document.createElement('caption');caption.textContent='Group-balanced mean absolute error; lower is better.';table.append(caption);
  const head=document.createElement('thead'),tr=document.createElement('tr');
  for(const name of ['Model','Error (percentage points)']){const th=document.createElement('th');th.scope='col';th.textContent=name;tr.append(th);}head.append(tr);table.append(head);
  const body=document.createElement('tbody');
  for(const name of ['Change','Appearance','Segmentation','Combined']){
   const row=document.createElement('tr'),label=document.createElement('th'),value=document.createElement('td');label.scope='row';label.textContent=name==='Change'?'Image-change baseline':name;value.textContent=data.experiment.summary[name].all.group_mae_pp.toFixed(2);row.append(label,value);body.append(row);
  }table.append(body);$('experiment-results').append(table);$('lab').hidden=false;
  if(!$('load-status').classList.contains('error'))$('load-status').hidden=true;
  const requested=location.hash.slice(1);if(ids.has(requested))$('filter').value='all';filter(requested);
 }catch(error){message('load-status',`${error.message||'Could not load the review lab.'} Reload after restoring the local evidence.`,true);}
}
for(const kind of ['before','after'])$(kind+'-photo').addEventListener('error',()=>message('image-status','Image unavailable. Turn off experimental masks or restore the local photos and mask cache.',true));
$('source-details').addEventListener('toggle',()=>{sourceViewed=sourceViewed||$('source-details').open;});
$('ai-details').addEventListener('toggle',()=>{proposalViewed=proposalViewed||$('ai-details').open;});
$('show-overlay').addEventListener('change',images);
$('filter').addEventListener('change',()=>filter(currentId));$('search').addEventListener('input',()=>filter(currentId));
for(const [button,offset] of [['previous',-1],['next',1]])$(button).addEventListener('click',()=>{const i=filtered.findIndex(r=>r.record_id===currentId);if(filtered[i+offset]){currentId=filtered[i+offset].record_id;render();}});
$('review-form').addEventListener('submit',event=>{
 event.preventDefault();if(!currentId||!$('review-form').reportValidity())return;
 const row=data.rows.find(r=>r.record_id===currentId);
 const prior=reviews[currentId];
 reviews[currentId]={record_id:currentId,before_observation:$('before-decision').value,observation:$('decision').value,pair_issue:$('pair-issue').checked,starting_portion_unverified:$('before-decision').value!=='visible_food',note:$('review-note').value.trim(),reviewed_at:new Date().toISOString(),reviewer:'local browser user (identity not verified)',source_labels_viewed:sourceViewed||!!prior?.source_labels_viewed,model_overlay_viewed:overlayViewed||!!prior?.model_overlay_viewed,ai_proposal_viewed:proposalViewed||!!prior?.ai_proposal_viewed,evidence_version:data.review_version,image_sha256:{before:row.images.before.sha256,after:row.images.after.sha256},mass_verified:false,evaluation_gold:false};
 try{localStorage.setItem(storageKey,JSON.stringify(reviews));message('save-status','Review saved. Source weights and model results are unchanged.');}
 catch{message('save-status','Saved in memory only. Export reviews before closing this page.',true);}
 progress();$('remove-review').hidden=false;$('pair-position').textContent=`Pair ${filtered.findIndex(r=>r.record_id===currentId)+1} of ${filtered.length} · Reviewed`;
});
$('remove-review').addEventListener('click',()=>{
 const updated={...reviews};delete updated[currentId];
 try{localStorage.setItem(storageKey,JSON.stringify(updated));reviews=updated;render();progress();message('save-status','Saved review removed. Original evidence is unchanged.');}
 catch{message('save-status','Could not update browser storage. The review was retained.',true);}
});
$('export').addEventListener('click',()=>{
 const payload={schema_version:2,kind:'visual_review_not_mass_ground_truth',evidence_version:data.review_version,exported_at:new Date().toISOString(),records:Object.values(reviews)};
 const text=JSON.stringify(payload,null,2)+'\n';
 if(exportUrl)URL.revokeObjectURL(exportUrl);
 exportUrl=URL.createObjectURL(new Blob([text],{type:'application/json'}));
 $('download-reviews').href=exportUrl;$('export-json').value=text;$('export-panel').hidden=false;$('export-json').focus();
});
$('close-export').addEventListener('click',()=>{$('export-panel').hidden=true;$('export').focus();});
start();
