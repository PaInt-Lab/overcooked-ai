// Mode Switcher Module - Handles switching between Edit Current Plan and Load Previous Plans
(function() {
  'use strict';

  document.addEventListener("DOMContentLoaded", function () {
    // Hide/show task section based on game state
    function hideTaskSection() {
      document.getElementById("task-section").style.display = "none";
    }

    function showTaskSection() {
      document.getElementById("task-section").style.display = "block";
    }

    // Create button - hide task section
    document
      .getElementById("create")
      .addEventListener("click", function () {
        hideTaskSection();
      });

    // Leave button - show task section
    document.getElementById("leave").addEventListener("click", function () {
      showTaskSection();
    });

    // Switch between Edit Current Plan and Load Previous Plans
    document.getElementById("editCurrentPlanBtn").addEventListener("click", function() {
      // Show edit current plan section
      document.getElementById("editCurrentPlanSection").style.display = "block";
      document.getElementById("loadPreviousPlansSection").style.display = "none";
      
      // Update button styles
      this.classList.add("active-mode");
      this.classList.remove("btn-outline-primary");
      this.classList.add("btn-primary");
      
      document.getElementById("loadPreviousPlansBtn").classList.remove("active-mode");
      document.getElementById("loadPreviousPlansBtn").classList.remove("btn-primary");
      document.getElementById("loadPreviousPlansBtn").classList.add("btn-outline-primary");
    });

    document.getElementById("loadPreviousPlansBtn").addEventListener("click", function() {
      // Show load previous plans section
      document.getElementById("editCurrentPlanSection").style.display = "none";
      document.getElementById("loadPreviousPlansSection").style.display = "block";
      
      // Update button styles
      this.classList.add("active-mode");
      this.classList.remove("btn-outline-primary");
      this.classList.add("btn-primary");
      
      document.getElementById("editCurrentPlanBtn").classList.remove("active-mode");
      document.getElementById("editCurrentPlanBtn").classList.remove("btn-primary");
      document.getElementById("editCurrentPlanBtn").classList.add("btn-outline-primary");
    });
  });
})();

