const $ = id => document.getElementById(id);
let current = null, originals = {}, uploaded = {}, highlights = null;
async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed.');
  return data;
}
function show(data) {
  $('estimate').textContent = `${(data.estimated_fraction*100).toFixed(1)}% estimated remaining`;
  $('reference').textContent=Number.isFinite(data.recorded_fraction) ? `Recorded: ${(data.recorded_fraction*100).toFixed(1)}%. Absolute error: ${(100*Math.abs(data.estimated_fraction-data.recorded_fraction)).toFixed(1)} percentage points.` : 'Accuracy on this pair is unknown without weighing.';
}
function photos() {
  for (const kind of ['before','after']) $(kind).src = originals[kind] || '';
  
}
async function choose(example) {
  current = example; highlights = null; originals = {before:example.before,after:example.after}; photos();
  try { show(await request('/api/predict',{example_id:example.id})); $('status').textContent=''; }
  catch (error) { $('status').textContent=error.message; }
}
async function start() {
  
  const samples = await request('/api/examples');
  for (const example of samples) {
    const button=document.createElement('button'); button.textContent=example.id;
    button.addEventListener('click',()=>choose(example)); $('examples').append(button);
  }
  await choose(samples[0]);
  const report=await request('/api/benchmark');
  for(const [collection,metrics] of Object.entries(report.summary)) {
    const title=document.createElement('h3');title.textContent=collection;$('benchmark').append(title);
    for(const [method,error] of Object.entries(metrics.mae_percentage_points)) {
      const line=document.createElement('p');line.textContent=`${method}: ${error.toFixed(2)} pp`;
      const bar=document.createElement('progress');bar.max=100;bar.value=error;line.append(bar);$('benchmark').append(line);
    }
  }
}
start().catch(error=>$('status').textContent=error.message);
