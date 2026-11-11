'use strict';
(async()=>{try{const response=await fetch('/project-status.json',{cache:'no-store'});if(!response.ok)throw Error('Status file not found');const plan=await response.json();document.getElementById('summary').textContent=plan.summary;document.getElementById('updated').textContent='Updated '+plan.updated;
for(const stage of plan.stages){const row=document.createElement('tr');row.dataset.current=stage.number===plan.current_stage;for(const value of [stage.number,stage.name,stage.status,stage.evidence]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}document.getElementById('stages').append(row);}
for(const gate of plan.gates){const item=document.createElement('p'),title=document.createElement('strong');title.textContent=`${gate.name}: ${gate.status}. `;item.append(title,document.createTextNode(gate.condition));document.getElementById('gates').append(item);}
}catch(error){document.getElementById('summary').textContent=error.message;}})();
