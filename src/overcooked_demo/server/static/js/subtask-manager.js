// Subtask Management Module
(function () {
  "use strict";

  let draggedElement = null;
  let isConfirmed = false;
  let isPlanLoading = false;

  document.addEventListener("DOMContentLoaded", function () {
    const container = document.getElementById("subtaskContainer");
    const confirmContainer = document.getElementById(
      "confirmSequenceContainer"
    );
    const confirmBtn = document.getElementById("confirmSequence");

    // Create subtask functionality
    function createSubtaskCard(text) {
      const emptyMessage = document.getElementById("emptyMessage");

      // Hide empty message
      if (emptyMessage) {
        emptyMessage.style.display = "none";
      }

      const card = document.createElement("div");
      card.className = "subtask-card";
      card.draggable = true;

      const subtaskNumber =
        container.querySelectorAll(".subtask-card").length + 1;
      card.innerHTML = `
        <span class="subtask-number">${subtaskNumber}</span>
        <span class="subtask-text">${text}</span>
        <button class="subtask-edit" onclick="editSubtask(this)" title="Edit">&#9998;</button>
        <button class="subtask-delete" onclick="removeSubtask(this)">&times;</button>
      `;

      // Add drag event listeners
      card.addEventListener("dragstart", function (e) {
        draggedElement = this;
        this.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
      });

      card.addEventListener("dragend", function (e) {
        this.classList.remove("dragging");
        draggedElement = null;
        updateSubtaskNumbers();
      });

      container.appendChild(card);
      updateConfirmButtonVisibility();
    }

    function updateSubtaskNumbers() {
      const cards = container.querySelectorAll(".subtask-card");
      cards.forEach((card, index) => {
        const numberSpan = card.querySelector(".subtask-number");
        numberSpan.textContent = index + 1;
      });
    }

    function removeSubtask(button) {
      const card = button.parentElement;
      card.remove();

      // Update numbering after removal
      updateSubtaskNumbers();

      // Show empty message if no cards left
      const cards = container.querySelectorAll(".subtask-card");
      if (cards.length === 0) {
        const empty = document.getElementById("emptyMessage");
        if (empty) {
          empty.style.display = "block";
        }
      }
      updateConfirmButtonVisibility();
    }

    // Expose removeSubtask to global scope so inline onclick can call it
    window.removeSubtask = removeSubtask;

    function editSubtask(button) {
      const card = button.parentElement;
      const textSpan = card.querySelector(".subtask-text");
      const oldText = textSpan.textContent.trim();

      // Replace the <span> with an <input>
      const input = document.createElement("input");
      input.type = "text";
      input.className = "subtask-edit-input";
      input.value = oldText;
      input.style.flexGrow = "1";
      input.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          saveEdit();
        }
      });

      function saveEdit() {
        const newText = input.value.trim();
        if (newText) {
          textSpan.textContent = newText;
        }
        card.replaceChild(textSpan, input);
      }

      // Swap span → input, then focus
      card.replaceChild(input, textSpan);
      input.focus();

      // When input loses focus, save
      input.addEventListener("blur", saveEdit);
    }

    // Expose to global:
    window.editSubtask = editSubtask;

    // Create subtask button
    document
      .getElementById("createSubtaskBtn")
      .addEventListener("click", function (e) {
        e.preventDefault();
        const input = document.getElementById("subtaskInput");
        const text = input.value.trim();
        if (text) {
          createSubtaskCard(text);
          input.value = "";
        }
      });

    // Enter key in input
    document
      .getElementById("subtaskInput")
      .addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          document.getElementById("createSubtaskBtn").click();
        }
      });

    // Ask LLM functionality
    document
      .getElementById("askLlmBtn")
      .addEventListener("click", async function (e) {
        e.preventDefault();
        const taskTitle = document.getElementById("taskTitle").value.trim();
        if (!taskTitle) {
          alert("Please enter a task title first");
          return;
        }

        // Gather existing subtasks from cards
        const cards = container.querySelectorAll(".subtask-card");
        const existingSubtasks = Array.from(cards).map((card) => {
          return card.querySelector(".subtask-text").textContent.trim();
        });

        // Disable button while waiting
        const btn = document.getElementById("askLlmBtn");
        btn.disabled = true;
        btn.textContent = "Generating…";

        const payload = {
          taskName: taskTitle,
          existingSubtasks: existingSubtasks,
          notes: "", // adjust if you add a notes field
        };

        try {
          const resp = await fetch("/generate_subtasks", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          });

          if (!resp.ok) {
            throw new Error(`Server returned ${resp.status}`);
          }

          const data = await resp.json();
          if (data.error) {
            throw new Error(data.error);
          }

          const subtasks = data.subtasks || [];

          // Hide the "No subtasks yet" message if it exists, then clear the container
          const empty = document.getElementById("emptyMessage");
          if (empty) {
            empty.style.display = "none";
          }
          container.innerHTML = "";

          // Populate new cards
          subtasks.forEach((subtask) => {
            createSubtaskCard(subtask);
          });
        } catch (err) {
          console.error("Error generating subtasks:", err);
          alert("Failed to generate subtasks: " + err.message);
        } finally {
          btn.disabled = false;
          btn.textContent = "Ask LLM";
        }
      });

    // Make Default Onion Recipe functionality
    document
      .getElementById("makeDefaultOnionRecipeBtn")
      .addEventListener("click", function (e) {
        e.preventDefault();

        // Clear existing subtasks
        const empty = document.getElementById("emptyMessage");
        if (empty) {
          empty.style.display = "none";
        }
        container.innerHTML = "";

        // Create cards for each recipe subtask using the default onion recipe from constants
        DEFAULT_SEASONED_PLAN.forEach((subtask) => {
          createSubtaskCard(subtask);
        });
      });

    // Make Default Tomato Recipe functionality
    document
      .getElementById("makeDefaultTomatoRecipeBtn")
      .addEventListener("click", function (e) {
        e.preventDefault();

        // Clear existing subtasks
        const empty = document.getElementById("emptyMessage");
        if (empty) {
          empty.style.display = "none";
        }
        container.innerHTML = "";

        // Create cards for each recipe subtask using the default tomato recipe from constants
        DEFAULT_TOMATO_PLAN.forEach((subtask) => {
          createSubtaskCard(subtask);
        });
      });

    // Make Mixed Onion Recipe functionality
    document
      .getElementById("makeMixedOnionRecipeBtn")
      .addEventListener("click", function (e) {
        e.preventDefault();

        // Clear existing subtasks
        const empty = document.getElementById("emptyMessage");
        if (empty) {
          empty.style.display = "none";
        }
        container.innerHTML = "";

        // Create cards for each recipe subtask using the mixed onion recipe from constants
        MIXED_ONION_FIRST_PLAN.forEach((subtask) => {
          createSubtaskCard(subtask);
        });
      });

    // Make Mixed Tomato Recipe functionality
    document
      .getElementById("makeMixedTomatoRecipeBtn")
      .addEventListener("click", function (e) {
        e.preventDefault();

        // Clear existing subtasks
        const empty = document.getElementById("emptyMessage");
        if (empty) {
          empty.style.display = "none";
        }
        container.innerHTML = "";

        // Create cards for each recipe subtask using the mixed tomato recipe from constants
        MIXED_TOMATO_FIRST_PLAN.forEach((subtask) => {
          createSubtaskCard(subtask);
        });
      });

    // Drag and drop for reordering
    container.addEventListener("dragover", function (e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      this.classList.add("drag-over");
    });

    container.addEventListener("dragleave", function (e) {
      this.classList.remove("drag-over");
    });

    container.addEventListener("drop", function (e) {
      e.preventDefault();
      this.classList.remove("drag-over");

      if (draggedElement) {
        const afterElement = getDragAfterElement(this, e.clientX, e.clientY);

        if (afterElement == null) {
          this.appendChild(draggedElement);
        } else {
          this.insertBefore(draggedElement, afterElement);
        }

        // Update numbering after reordering
        updateSubtaskNumbers();
      }
    });

    function getDragAfterElement(container, x, y) {
      const draggableElements = [
        ...container.querySelectorAll(".subtask-card:not(.dragging)"),
      ];

      return draggableElements.reduce(
        (closest, child) => {
          const box = child.getBoundingClientRect();
          const offset = y - box.top - box.height / 2;

          if (offset < 0 && offset > closest.offset) {
            return { offset: offset, element: child };
          } else {
            return closest;
          }
        },
        { offset: Number.NEGATIVE_INFINITY }
      ).element;
    }

    function updateConfirmButtonVisibility() {
      confirmContainer.style.display = container.querySelectorAll(
        ".subtask-card"
      ).length
        ? "block"
        : "none";
    }

    function toggleConfirm() {
      if (isConfirmed) {
        // If already confirmed, just toggle back to edit mode
        isConfirmed = false;
        document.querySelectorAll(".subtask-card").forEach((card) => {
          card.classList.remove("confirmed");
          card.draggable = true;
        });
        confirmBtn.textContent = "Confirm Sequence";
        document.getElementById("planLoadingContainer").style.display = "none";
        document.getElementById("planCreationOverlay").style.display = "none";
        confirmBtn.disabled = false;
        return;
      }

      // Show loading state
      isPlanLoading = true;
      const planLoadingContainer = document.getElementById(
        "planLoadingContainer"
      );
      const planCreationOverlay = document.getElementById(
        "planCreationOverlay"
      );
      planLoadingContainer.style.display = "block";
      planCreationOverlay.style.display = "flex";
      confirmBtn.disabled = true;
      confirmBtn.textContent = "Creating Plan...";

      const seq = [...container.querySelectorAll(".subtask-card")].map((c) =>
        c.querySelector(".subtask-text").textContent.trim()
      );

      // Get the task title, time, and day from input fields
      const taskTitle = document.getElementById("taskTitle").value.trim();
      const planTime = document.getElementById("planTime").value;
      const planDay = document.getElementById("planDay").value;

      fetch("/confirm_subtasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subtasks: seq,
          taskTitle: taskTitle,
          planTime: planTime,
          planDay: planDay,
        }),
      })
        .then((r) => r.json())
        .then((data) => {
          console.log("confirm_subtasks response:", data);
          if (data.status === "success") {
            window.planSessionId = data.session_id;
            window.taskTitle = data.taskTitle; // Store task title globally
            window.planTime = data.planTime; // Store plan time globally
            window.planDay = data.planDay; // Store plan day globally

            // Debug: Print initial user plan
            console.log("=== INITIAL USER PLAN (DEBUG) ===");
            console.log("Task Title:", data.taskTitle);
            console.log("Plan Time:", data.planTime);
            console.log("Plan Day:", data.planDay);
            console.log("Session ID:", data.session_id);
            console.log("Subtasks:");
            seq.forEach((subtask, index) => {
              console.log(`  ${index + 1}. ${subtask}`);
            });
            console.log("=================================");

            isConfirmed = true;
            document.querySelectorAll(".subtask-card").forEach((card) => {
              card.classList.add("confirmed");
              card.draggable = false;
            });
            confirmBtn.textContent = "Edit Sequence";
            planLoadingContainer.style.display = "none";
            document.getElementById("planCreationOverlay").style.display =
              "none";
            confirmBtn.disabled = false;
            isPlanLoading = false;
          } else {
            throw new Error(data.error || "Failed to create plan");
          }
        })
        .catch((error) => {
          console.error("Error creating plan:", error);
          alert("Failed to create plan: " + error.message);
          // Reset to original state on error
          planLoadingContainer.style.display = "none";
          document.getElementById("planCreationOverlay").style.display = "none";
          confirmBtn.disabled = false;
          confirmBtn.textContent = "Confirm Sequence";
          isPlanLoading = false;
        });
    }
    confirmBtn.addEventListener("click", toggleConfirm);

    // No Subtasks button — skips subtask pipeline, sends empty subtasks
    document
      .getElementById("noSubtasksBtn")
      .addEventListener("click", async function (e) {
        e.preventDefault();
        const taskTitle = document.getElementById("taskTitle").value.trim();
        if (!taskTitle) {
          alert("Please enter a task title first");
          return;
        }
        if (isConfirmed) {
          // Toggle back to edit mode
          isConfirmed = false;
          this.textContent = "No Subtasks";
          this.classList.remove("btn-secondary");
          this.classList.add("btn-light-red");
          return;
        }

        this.disabled = true;
        this.textContent = "Setting up...";

        const planTime = document.getElementById("planTime").value;
        const planDay = document.getElementById("planDay").value;

        try {
          const resp = await fetch("/confirm_subtasks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              subtasks: [],
              taskTitle: taskTitle,
              planTime: planTime,
              planDay: planDay,
            }),
          });
          const data = await resp.json();
          if (data.status === "success") {
            window.planSessionId = data.session_id;
            window.taskTitle = data.taskTitle;
            window.planTime = data.planTime;
            window.planDay = data.planDay;
            isConfirmed = true;
            this.textContent = "No Subtasks ✓";
            this.classList.remove("btn-light-red");
            this.classList.add("btn-secondary");
          } else {
            throw new Error(data.error || "Failed to create plan");
          }
        } catch (err) {
          alert("Failed: " + err.message);
          this.textContent = "No Subtasks";
        } finally {
          this.disabled = false;
        }
      });

    // Make time picker open when clicking anywhere on the input
    const planTimeInput = document.getElementById("planTime");
    planTimeInput.addEventListener("click", function () {
      try {
        // Show the time picker dropdown
        if (this.showPicker) {
          this.showPicker();
        }
      } catch (e) {
        // showPicker() not supported in some browsers, ignore
      }
    });

    // Create game button validation
    const createBtn = document.getElementById("create");
    createBtn.addEventListener(
      "click",
      (e) => {
        const p0 = document.getElementById("playerZero").value;
        const p1 = document.getElementById("playerOne").value;
        const llmSelected = p0 === "overcooked_llm" || p1 === "overcooked_llm";

        // Check if plan is currently being created
        if (isPlanLoading) {
          e.preventDefault();
          e.stopImmediatePropagation();
          alert(
            "Please wait for the plan to be created before starting the game."
          );
          return;
        }

        if (llmSelected && !isConfirmed) {
          e.preventDefault();
          e.stopImmediatePropagation();
          alert(
            "Please confirm your subtask sequence before using the overcooked_llm agent."
          );
          return;
        }
      },
      true
    );
  });
})();
