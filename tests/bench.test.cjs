// Check the actual bundled program and offline prediction controls in a DOM stub.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root,'index.html'), 'utf8');
const fixture = require('./upstream_reference.json');
const elements = new Map();
const pen = { fillRect(){}, fillText(){}, strokeRect(){} };
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: id==='head'?'L1H1':'', width:700, height:360,
    getContext(){return pen}, replaceChildren(){}, append(){}
  });
  return elements.get(id);
}
const context = {document:{getElementById: element, createElement:()=>({append(){}})}};
vm.createContext(context);
const run = code => vm.runInContext(code, context);
run(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
assert.equal(element('status').textContent, 'Sealed');
element('edit').onclick();
assert.equal(element('input').value.split(' ')[1], 'moon');
for (let h=0;h<2;h++) {
  element('head').value='L1H'+(h+1);
  element('input').value=fixture.words.join(' ');
  element('head').onchange();
  element('reveal').onclick();
  assert.equal(element('mapTitle').textContent, 'Actual routing');
  for (const [i, expected] of fixture.targets[h].attention.entries()) {
    assert.ok(Math.abs(run('truth.at(-1)')[i]-expected)<1e-12);
    assert.ok(Math.abs(run('prediction.at(-1)')[i]-expected)<1e-10);
  }
  element('rule').value='lag';
  element('rule').onchange();
  assert.ok(run('prediction.at(-1)').some((w,i)=>Math.abs(w-fixture.targets[h].attention[i])>.01));
  element('rule').value='full';
  element('rule').onchange();
  element('toggle').onclick();
  element('toggle').onclick();
  assert.equal(element('mapTitle').textContent,'Predicted routing');
  element('position').value='1';
  element('position').oninput();
  assert.match(element('posLabel').textContent,/word 1 /);
}
element('input').value='oops';
element('predict').onclick();
assert.match(element('message').textContent,/Unknown word/);
console.log('Bundled controls and both independent upstream attention rows passed.');
