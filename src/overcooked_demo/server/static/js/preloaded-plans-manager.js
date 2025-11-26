// Preloaded Plans Management Module
(function() {
  'use strict';

  document.addEventListener("DOMContentLoaded", function () {
    // Create a preloaded plan card
    function createPreloadedPlanCard(planId, subtasks = [], planTime = "14:00", planDay = "Wednesday") {
      const container = document.getElementById("preloadedPlansContainer");
      const noMessage = document.getElementById("noPreloadedPlansMessage");
      
      if (noMessage) {
        noMessage.style.display = "none";
      }

      const card = document.createElement("div");
      card.className = "preloaded-plan-card";
      card.setAttribute("data-plan-id", planId);

      card.innerHTML = `
        <div class="plan-header" data-plan-id="${planId}">
          <div>
            <span class="plan-title">
              <span class="collapse-icon">▼</span>
              Plan #${planId}
            </span>
          </div>
          <div class="plan-time-inputs">
            <input type="time" class="plan-time-input" value="${planTime}" data-plan-id="${planId}" onclick="event.stopPropagation()">
            <select class="plan-day-select" data-plan-id="${planId}" onclick="event.stopPropagation()">
              <option value="Monday" ${planDay === "Monday" ? "selected" : ""}>Monday</option>
              <option value="Tuesday" ${planDay === "Tuesday" ? "selected" : ""}>Tuesday</option>
              <option value="Wednesday" ${planDay === "Wednesday" ? "selected" : ""}>Wednesday</option>
              <option value="Thursday" ${planDay === "Thursday" ? "selected" : ""}>Thursday</option>
              <option value="Friday" ${planDay === "Friday" ? "selected" : ""}>Friday</option>
              <option value="Saturday" ${planDay === "Saturday" ? "selected" : ""}>Saturday</option>
              <option value="Sunday" ${planDay === "Sunday" ? "selected" : ""}>Sunday</option>
            </select>
            <button class="delete-plan-btn" onclick="event.stopPropagation(); deletePreloadedPlan(${planId})">Delete Plan</button>
          </div>
        </div>
        <div class="plan-content" data-plan-id="${planId}">
          <div class="plan-subtasks-container" data-plan-id="${planId}">
            <p class="text-muted text-center empty-subtasks-message">No subtasks yet. Add some below!</p>
          </div>
          <input type="text" class="form-control mt-2 subtask-input-field" placeholder="Enter subtask..." data-plan-id="${planId}">
          <button class="add-preloaded-subtask-btn" data-plan-id="${planId}">Add Subtask</button>
        </div>
      `;

      container.appendChild(card);

      // Add collapse/expand functionality
      const header = card.querySelector('.plan-header');
      const content = card.querySelector('.plan-content');
      const collapseIcon = card.querySelector('.collapse-icon');
      
      header.addEventListener('click', function(e) {
        // Don't collapse if clicking on inputs/buttons (they have stopPropagation)
        if (content.classList.contains('collapsed')) {
          content.classList.remove('collapsed');
          collapseIcon.classList.remove('collapsed');
        } else {
          content.classList.add('collapsed');
          collapseIcon.classList.add('collapsed');
        }
      });

      // Add existing subtasks if provided
      if (subtasks.length > 0) {
        const subtasksContainer = card.querySelector('.plan-subtasks-container');
        const emptyMsg = subtasksContainer.querySelector('.empty-subtasks-message');
        if (emptyMsg) emptyMsg.style.display = "none";
        
        subtasks.forEach(subtask => {
          createPreloadedSubtaskCard(planId, subtask);
        });
      }

      // Event listener for time input
      card.querySelector('.plan-time-input').addEventListener('change', function() {
        updatePreloadedPlanData(planId);
      });

      // Event listener for day select
      card.querySelector('.plan-day-select').addEventListener('change', function() {
        updatePreloadedPlanData(planId);
      });

      // Event listener for add subtask button
      card.querySelector('.add-preloaded-subtask-btn').addEventListener('click', function() {
        const input = card.querySelector('.subtask-input-field');
        const text = input.value.trim();
        if (text) {
          createPreloadedSubtaskCard(planId, text);
          input.value = "";
          updatePreloadedPlanData(planId);
        }
      });

      // Event listener for enter key in subtask input
      card.querySelector('.subtask-input-field').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
          e.preventDefault();
          card.querySelector('.add-preloaded-subtask-btn').click();
        }
      });

      // Setup drag and drop
      setupPlanDragAndDrop(planId);

      return card;
    }

    // Create a subtask card within a preloaded plan
    function createPreloadedSubtaskCard(planId, text) {
      const container = document.querySelector(`.plan-subtasks-container[data-plan-id="${planId}"]`);
      const emptyMsg = container.querySelector('.empty-subtasks-message');
      if (emptyMsg) {
        emptyMsg.style.display = "none";
      }

      const card = document.createElement("div");
      card.className = "preloaded-subtask-card";
      card.draggable = true;

      const subtaskNumber = container.querySelectorAll(".preloaded-subtask-card").length + 1;
      card.innerHTML = `
        <span class="subtask-number">${subtaskNumber}</span>
        <span class="subtask-text">${text}</span>
        <button class="subtask-edit" onclick="editPreloadedSubtask(this, ${planId})" title="Edit">&#9998;</button>
        <button class="subtask-delete" onclick="removePreloadedSubtask(this, ${planId})">&times;</button>
      `;

      // Drag event listeners
      card.addEventListener("dragstart", function(e) {
        this.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
      });

      card.addEventListener("dragend", function(e) {
        this.classList.remove("dragging");
        updatePreloadedSubtaskNumbers(planId);
        updatePreloadedPlanData(planId);
      });

      container.appendChild(card);
    }

    // Setup drag and drop for a plan's subtasks
    function setupPlanDragAndDrop(planId) {
      const container = document.querySelector(`.plan-subtasks-container[data-plan-id="${planId}"]`);

      container.addEventListener("dragover", function(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        this.classList.add("drag-over");
      });

      container.addEventListener("dragleave", function(e) {
        this.classList.remove("drag-over");
      });

      container.addEventListener("drop", function(e) {
        e.preventDefault();
        this.classList.remove("drag-over");

        const dragging = this.querySelector(".dragging");
        if (dragging) {
          const afterElement = getDragAfterElementInPlan(this, e.clientY);
          if (afterElement == null) {
            this.appendChild(dragging);
          } else {
            this.insertBefore(dragging, afterElement);
          }
          updatePreloadedSubtaskNumbers(planId);
          updatePreloadedPlanData(planId);
        }
      });
    }

    function getDragAfterElementInPlan(container, y) {
      const draggableElements = [...container.querySelectorAll(".preloaded-subtask-card:not(.dragging)")];

      return draggableElements.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;

        if (offset < 0 && offset > closest.offset) {
          return { offset: offset, element: child };
        } else {
          return closest;
        }
      }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    // Update subtask numbers for a plan
    function updatePreloadedSubtaskNumbers(planId) {
      const container = document.querySelector(`.plan-subtasks-container[data-plan-id="${planId}"]`);
      const cards = container.querySelectorAll(".preloaded-subtask-card");
      cards.forEach((card, index) => {
        const numberSpan = card.querySelector(".subtask-number");
        numberSpan.textContent = index + 1;
      });
    }

    // Update preloaded plan data in window.preloadedPlans
    function updatePreloadedPlanData(planId) {
      const card = document.querySelector(`.preloaded-plan-card[data-plan-id="${planId}"]`);
      const timeInput = card.querySelector('.plan-time-input');
      const daySelect = card.querySelector('.plan-day-select');
      const subtasksContainer = card.querySelector('.plan-subtasks-container');
      const subtaskCards = subtasksContainer.querySelectorAll('.preloaded-subtask-card');

      const subtasks = Array.from(subtaskCards).map(card => 
        card.querySelector('.subtask-text').textContent.trim()
      );

      // Find or create plan in window.preloadedPlans
      const existingPlanIndex = window.preloadedPlans.findIndex(p => p.id === planId);
      const planData = {
        id: planId,
        subtasks: subtasks,
        planTime: timeInput.value,
        planDay: daySelect.value
      };

      if (existingPlanIndex >= 0) {
        window.preloadedPlans[existingPlanIndex] = planData;
      } else {
        window.preloadedPlans.push(planData);
      }
    }

    // Remove a preloaded subtask
    window.removePreloadedSubtask = function(button, planId) {
      const card = button.parentElement;
      card.remove();
      updatePreloadedSubtaskNumbers(planId);
      updatePreloadedPlanData(planId);

      // Show empty message if no cards left
      const container = document.querySelector(`.plan-subtasks-container[data-plan-id="${planId}"]`);
      const cards = container.querySelectorAll(".preloaded-subtask-card");
      if (cards.length === 0) {
        const emptyMsg = container.querySelector('.empty-subtasks-message');
        if (emptyMsg) {
          emptyMsg.style.display = "block";
        }
      }
    };

    // Edit a preloaded subtask
    window.editPreloadedSubtask = function(button, planId) {
      const card = button.parentElement;
      const textSpan = card.querySelector(".subtask-text");
      const oldText = textSpan.textContent.trim();

      const input = document.createElement("input");
      input.type = "text";
      input.className = "subtask-edit-input";
      input.value = oldText;
      input.style.flexGrow = "1";

      function saveEdit() {
        const newText = input.value.trim();
        if (newText) {
          textSpan.textContent = newText;
          updatePreloadedPlanData(planId);
        }
        card.replaceChild(textSpan, input);
      }

      input.addEventListener("keypress", function(e) {
        if (e.key === "Enter") {
          e.preventDefault();
          saveEdit();
        }
      });

      input.addEventListener("blur", saveEdit);

      card.replaceChild(input, textSpan);
      input.focus();
    };

    // Delete a preloaded plan
    window.deletePreloadedPlan = function(planId) {
      const card = document.querySelector(`.preloaded-plan-card[data-plan-id="${planId}"]`);
      card.remove();

      // Remove from window.preloadedPlans
      window.preloadedPlans = window.preloadedPlans.filter(p => p.id !== planId);

      // Show empty message if no plans left
      const container = document.getElementById("preloadedPlansContainer");
      const cards = container.querySelectorAll(".preloaded-plan-card");
      if (cards.length === 0) {
        const noMessage = document.getElementById("noPreloadedPlansMessage");
        if (noMessage) {
          noMessage.style.display = "block";
        }
      }
    };

    // Add default plan button
    document.getElementById("addDefaultPlanBtn").addEventListener("click", function() {
      // Use current count + 1 for plan ID so numbering restarts from 1 when all plans are deleted
      const container = document.getElementById("preloadedPlansContainer");
      const existingPlans = container.querySelectorAll(".preloaded-plan-card");
      const planId = existingPlans.length + 1;
      
      createPreloadedPlanCard(planId, DEFAULT_SEASONED_PLAN, "14:00", "Wednesday");
      updatePreloadedPlanData(planId);
    });

    // Add tomato plan button
    document.getElementById("addTomatoPlanBtn").addEventListener("click", function() {
      // Use current count + 1 for plan ID so numbering restarts from 1 when all plans are deleted
      const container = document.getElementById("preloadedPlansContainer");
      const existingPlans = container.querySelectorAll(".preloaded-plan-card");
      const planId = existingPlans.length + 1;
      
      createPreloadedPlanCard(planId, DEFAULT_TOMATO_PLAN, "14:00", "Wednesday");
      updatePreloadedPlanData(planId);
    });

    // Add mixed onion plan button
    document.getElementById("addMixedOnionPlanBtn").addEventListener("click", function() {
      // Use current count + 1 for plan ID so numbering restarts from 1 when all plans are deleted
      const container = document.getElementById("preloadedPlansContainer");
      const existingPlans = container.querySelectorAll(".preloaded-plan-card");
      const planId = existingPlans.length + 1;
      
      createPreloadedPlanCard(planId, MIXED_ONION_FIRST_PLAN, "14:00", "Wednesday");
      updatePreloadedPlanData(planId);
    });

    // Add mixed tomato plan button
    document.getElementById("addMixedTomatoPlanBtn").addEventListener("click", function() {
      // Use current count + 1 for plan ID so numbering restarts from 1 when all plans are deleted
      const container = document.getElementById("preloadedPlansContainer");
      const existingPlans = container.querySelectorAll(".preloaded-plan-card");
      const planId = existingPlans.length + 1;
      
      createPreloadedPlanCard(planId, MIXED_TOMATO_FIRST_PLAN, "14:00", "Wednesday");
      updatePreloadedPlanData(planId);
    });

    // Add custom plan button
    document.getElementById("addCustomPlanBtn").addEventListener("click", function() {
      // Use current count + 1 for plan ID so numbering restarts from 1 when all plans are deleted
      const container = document.getElementById("preloadedPlansContainer");
      const existingPlans = container.querySelectorAll(".preloaded-plan-card");
      const planId = existingPlans.length + 1;
      
      createPreloadedPlanCard(planId, [], "12:00", "Monday");
      updatePreloadedPlanData(planId);
    });
  });
})();

