const API = window.location.origin;
const $ = (id) => document.getElementById(id);
let selectedFile = null;
let lastResult = null;

const fieldDefinitions = [
  ["full_name","Full name"],["document_number","Document number"],["course","Course"],
  ["date_of_birth","Date of birth"],["issue_date","Issue date"],["expiry_date","Expiry date"],
  ["blood_group","Blood group"],["emergency_number","Emergency number"]
];

async function checkApi(){
  try {
    const res=await fetch(`${API}/`);
    if(!res.ok) throw new Error("API not ready");
    $("api-status").textContent="Connected to FastAPI";
  } catch(e) {
    $("api-status").textContent="API unavailable";
    showToast("Start the API with: python -m uvicorn main:app --reload");
  }
}
function showToast(message){$("toast").textContent=message;$("toast").classList.remove("hidden");setTimeout(()=>$("toast").classList.add("hidden"),4200);}
function showError(message){$("error-box").textContent=message;$("error-box").classList.remove("hidden");}
function clearError(){$("error-box").classList.add("hidden");$("error-box").textContent="";}
function formatSize(n){return n<1024*1024?`${(n/1024).toFixed(0)} KB`:`${(n/1024/1024).toFixed(2)} MB`;}
function chooseFile(file){
  clearError();
  if(!file)return;
  if(!["image/jpeg","image/png","image/webp"].includes(file.type)){showError("Please choose a JPG, PNG, or WEBP image.");return;}
  if(file.size>12*1024*1024){showError("This image is larger than 12 MB. Choose a smaller image.");return;}
  selectedFile=file;
  $("file-name").textContent=file.name;$("file-size").textContent=formatSize(file.size);
  $("selected-file").classList.remove("hidden");$("preview-wrap").classList.remove("hidden");
  $("preview").src=URL.createObjectURL(file);$("process-btn").disabled=false;
}
$("browse-btn").addEventListener("click",()=> $("file-input").click());
$("dropzone").addEventListener("click",(e)=>{if(!e.target.closest("button"))$("file-input").click();});
$("dropzone").addEventListener("keydown",(e)=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();$("file-input").click();}});
$("file-input").addEventListener("change",e=>chooseFile(e.target.files[0]));
$("remove-file").addEventListener("click",()=>{selectedFile=null;$("file-input").value="";$("selected-file").classList.add("hidden");$("preview-wrap").classList.add("hidden");$("process-btn").disabled=true;clearError();});
["dragenter","dragover"].forEach(evt=>$("dropzone").addEventListener(evt,e=>{e.preventDefault();$("dropzone").classList.add("dragover");}));
["dragleave","drop"].forEach(evt=>$("dropzone").addEventListener(evt,e=>{e.preventDefault();$("dropzone").classList.remove("dragover");}));
$("dropzone").addEventListener("drop",e=>chooseFile(e.dataTransfer.files[0]));

function fileUrl(path){if(!path)return null;return `${API}/api/v1/files/${path.split("/").map(encodeURIComponent).join("/")}`;}
function renderFields(data){
  $("field-grid").innerHTML="";
  fieldDefinitions.forEach(([key,label])=>{
    const value=data[key];
    const item=document.createElement("div");item.className="field";
    const lab=document.createElement("label");lab.textContent=label;
    const val=document.createElement("strong");val.textContent=(value===null||value===undefined||value==="")?"Not detected":value;
    if(value===null||value===undefined||value==="")val.classList.add("missing");
    item.append(lab,val);$("field-grid").append(item);
  });
}
function setImage(imgId,missingId,path){
  const img=$(imgId), missing=$(missingId), url=fileUrl(path);
  if(url){img.src=url;img.classList.remove("hidden");missing.classList.add("hidden");img.onerror=()=>{img.classList.add("hidden");missing.classList.remove("hidden");missing.textContent="Image could not be loaded";};}
  else {img.classList.add("hidden");missing.classList.remove("hidden");}
}
$("process-btn").addEventListener("click",async()=>{
  if(!selectedFile)return;
  clearError();$("empty-state").classList.add("hidden");$("results").classList.add("hidden");$("loading-state").classList.remove("hidden");
  $("result-status").textContent="PROCESSING";$("process-btn").disabled=true;
  const fd=new FormData();fd.append("file",selectedFile);
  try{
    const response=await fetch(`${API}/api/v1/process-id`,{method:"POST",body:fd});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||`Request failed (${response.status})`);
    lastResult=data;$("loading-state").classList.add("hidden");$("results").classList.remove("hidden");
    $("result-status").textContent="PROCESS COMPLETE";$("record-id").textContent=`Record ID: ${data.record_id||"saved"}`;
    renderFields(data);
    $("face-status").textContent=data.face_image?"Detected":"Not detected";
    $("face-link").textContent=data.face_image?data.face_image:"No portrait crop was saved";
    $("card-status").textContent=data.card_boundary_detected?"Aligned":"Not detected";
    $("card-note").textContent=data.card_boundary_detected?"Perspective correction applied":"OCR used the original image";
    setImage("processed-image","processed-missing",data.processed_image);
    setImage("face-image","face-missing",data.face_image);
    $("ocr-text").textContent=data.ocr_text||"No OCR text returned.";
    showToast("ID card processed and record saved.");
  }catch(e){
    $("loading-state").classList.add("hidden");$("empty-state").classList.remove("hidden");
    $("result-status").textContent="PROCESSING FAILED";showError(e.message||"Could not connect to the API. Is the server running?");
  }finally{$("process-btn").disabled=!selectedFile;}
});
$("download-json").addEventListener("click",()=>{
  if(!lastResult)return;
  const blob=new Blob([JSON.stringify(lastResult,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=`id-scan-${lastResult.record_id||"result"}.json`;a.click();URL.revokeObjectURL(url);
});
function switchView(view){
  const scanner=view==="scanner";$("scanner-view").classList.toggle("hidden",!scanner);$("records-view").classList.toggle("hidden",scanner);
  $("page-title").textContent=scanner?"ID Card Scanner":"Scan History";
  document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b.dataset.view===view));
  if(!scanner)loadHistory();
}
document.querySelectorAll(".nav-item").forEach(b=>b.addEventListener("click",()=>switchView(b.dataset.view)));
$("refresh-records").addEventListener("click",loadHistory);
async function loadHistory(){
  $("history-loading").classList.remove("hidden");$("history-content").classList.add("hidden");
  try{
    const r=await fetch(`${API}/api/v1/records?limit=100`);const data=await r.json();if(!r.ok)throw new Error(data.detail||"Could not load history");
    const tbody=$("history-body");tbody.innerHTML="";
    (data.records||[]).forEach(rec=>{
      const tr=document.createElement("tr");
      [rec.full_name,rec.document_number,rec.course,rec.issue_date,rec.expiry_date,rec.created_at].forEach(v=>{const td=document.createElement("td");td.textContent=v||"—";tr.append(td);});
      const action=document.createElement("td");const btn=document.createElement("button");btn.className="secondary-button";btn.textContent="View";btn.addEventListener("click",()=>showRecord(rec.record_id));action.append(btn);tr.append(action);tbody.append(tr);
    });
    $("history-empty").classList.toggle("hidden",(data.records||[]).length>0);
    $("history-loading").classList.add("hidden");$("history-content").classList.remove("hidden");
  }catch(e){$("history-loading").textContent=`Could not load history: ${e.message}`;}
}
async function showRecord(id){
  try{
    const r=await fetch(`${API}/api/v1/records/${encodeURIComponent(id)}`);const data=await r.json();if(!r.ok)throw new Error(data.detail||"Record not found");
    lastResult=data;switchView("scanner");$("empty-state").classList.add("hidden");$("loading-state").classList.add("hidden");$("results").classList.remove("hidden");$("result-status").textContent="SAVED RECORD";$("record-id").textContent=`Record ID: ${data.record_id}`;renderFields(data);
    $("face-status").textContent=data.face_image?"Detected":"Not detected";$("face-link").textContent=data.face_image||"No portrait crop was saved";$("card-status").textContent=data.card_boundary_detected?"Aligned":"Not detected";$("card-note").textContent=data.card_boundary_detected?"Perspective correction applied":"Original image used";
    setImage("processed-image","processed-missing",data.processed_image);setImage("face-image","face-missing",data.face_image);$("ocr-text").textContent=data.ocr_text||"No OCR text returned.";
  }catch(e){showToast(e.message);}
}
checkApi();
