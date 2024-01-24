document.addEventListener('DOMContentLoaded', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      var activeTab = tabs[0];
      var fetchedURL = activeTab.url;
      // fetch('http://54.198.2.223:5000/run_python_code', {
      fetch('http://localhost:5000/run_python_code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: fetchedURL }),
        mode: 'cors',
      })
        .then(response => response.json())
        .then(data => {

          var resultText = document.getElementById('resultText');
          var resultText1 = document.getElementById('resultText1');
          var resultText2 = document.getElementById('resultText2');
          var resultText3 = document.getElementById('resultText3');
          var resultText4 = document.getElementById('resultText4');
          var resultText5 = document.getElementById('resultText5');
          var card = document.getElementById('card');
          if(data.result[0]!="No deception found")
           card.style.display="block";          
          resultText1.textContent=data.result[1];
          resultText.textContent = data.result[2]+data.result[0];
          resultText2.textContent=data.result[3];
          resultText3.textContent=data.result[4];
          resultText4.textContent=data.result[5];
          resultText5.textContent=data.result[6];
         
        })
        .catch(error => {
          console.error('Error:', error);
          var resultText = document.getElementById('resultText');
          // resultText.textContent = "Page is too busy, Retry after sometime.";
          resultText.textContent = error;
        });
    });
  });
  