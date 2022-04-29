const grabBtn1 = document.getElementById("grabBtn1");
grabBtn1.addEventListener("click",() => {    
    chrome.storage.sync.set({cvd_type: "proto"}, function() {
        console.log("proto");
    });
});

const grabBtn2 = document.getElementById("grabBtn2");
grabBtn2.addEventListener("click",() => {    
    chrome.storage.sync.set({cvd_type: "deuto"}, function() {
        console.log("deuto");
    });
});

const grabBtn3 = document.getElementById("grabBtn3");
grabBtn3.addEventListener("click",() => {    
    chrome.storage.sync.set({cvd_type: "trita"}, function() {
        console.log("trita");
    });
});


const enabledbtn = document.getElementById("enabled");
enabledbtn.addEventListener("click",() => {  
    chrome.storage.sync.set({
		enabled: document.getElementById("enabled").checked
	}, function() {
        console.log("value changed");		
	});
});


function restore() {
	chrome.storage.sync.get({
		enabled: false,
        cvd_type: "proto"		
	}, function(items) {
		document.getElementById("enabled").checked = items.enabled;
	});
}
document.addEventListener("DOMContentLoaded", restore);