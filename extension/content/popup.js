const grabBtn = document.getElementById("grabBtn");
grabBtn.addEventListener("click", () => {
  var opener = window.opener;
  if (opener) {
    var oDom = opener.document;
    var elem = oDom.getElementById("img");
    alert(elem);
    if (elem) {
      var val = elem.value;
    }
  }

  console.log(image_list);
  alert(image_list.length);
  //const execSync = require('child_process').execSync;
  $.ajax({
    type: "POST",
    data: { foo: image_list },
    url: "http://127.0.0.1:105/gen_cvd/",
    success: function (msg) {
      $(".answer").html(msg);
    },
    headers: { "Content-Type": "application/json" },
  });
});

function popup() {
  chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
    var activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, { message: "start" });
  });
}

document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("button1").addEventListener("click", popup);
});
