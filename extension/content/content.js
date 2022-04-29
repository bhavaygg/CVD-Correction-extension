"use strict";

var response = "";
var image_link = "";
var cvd_type = "";
var enabled = "";

chrome.storage.sync.get(['cvd_type'], 
    function(items) {
        if (items.cvd_type == "proto") {cvd_type = "proto";}
        else if (items.cvd_type == "deuto") {cvd_type = "deuto";}
        else if (items.cvd_type == "trita") {cvd_type = "trita";}
        else {cvd_type = "proto";};
        console.log("cvd type : "+cvd_type);
});

chrome.storage.sync.get(['enabled'], 
    function(data){
        if (data.enabled){enabled = "true";}
        else{enabled = "false";}
});

const server_link = "http://127.0.0.1:5000/process/";

console.log(cvd_type);
console.log(localStorage.getItem("cvd_type"));

window.onload = async function(){
    if (enabled == "true"){
        console.log("CVD viewer is running");           
        for (let i=0; i<document.images.length; i++){    
            if (document.images[i].src) {
                var full_link = server_link + "?"+ "&sequence=" + i +"&link=" + document.images[i].src  + "&cvd_type=" + cvd_type;            
                response = await fetch(full_link, {cache: "no-store"});
                response = await response.json();
                if (response.response==200){
                    image_link = "chrome-extension://kgndlmfamiddbfgdhldndindompcebjp/images/" + response.path;
                    document.images[i].srcset = image_link;  
                };
            };
        };
        console.log("CVD viewer closed");
    }
    else{console.log("CVD viewer is not running");}
};