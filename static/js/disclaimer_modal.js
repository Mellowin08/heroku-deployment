//Function to manage modal display
function manageModalDisplay(modalID, buttonID, cookieName) {
    const modal = document.getElementById(modalID);
    const understandButton = document.getElementById(buttonID);
    // Check if user has agreed to disclaimer
    function hasUserAgreed() {
        return document.cookie.includes(`${cookieName}=true`);
    } 
    //  Display modal if user has not agreed to disclaimer
    if (!hasUserAgreed()) {
        window.addEventListener("load", function () {
            modal.style.display = "block";
        });
    }
    // Set cookie and hide modal if user has agreed to disclaimer
    understandButton.addEventListener("click", function () {
        document.cookie = `${cookieName}=true; expires=Fri, 31 Dec 9999 23:59:59 PHT`;
        modal.style.display = "none";
    });
}


