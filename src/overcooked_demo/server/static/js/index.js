// Persistent network connection that will be used to transmit real-time data
var socket = io();

/* * * * * * * * * * * * * * * *
 * Button click event handlers *
 * * * * * * * * * * * * * * * */

$(function () {
  $("#create").click(function () {
    params = arrToJSON($("form").serializeArray());
    params.layouts = [params.layout];
    data = {
      params: params,
      game_name: "overcooked",
      plan_session_id: window.planSessionId,
      preloaded_plans: window.preloadedPlans || [],
      create_if_not_found: false,
    };
    socket.emit("create", data);
    $("#waiting").show();
    $("#join").hide();
    $("#join").attr("disabled", true);
    $("#create").hide();
    $("#create").attr("disabled", true);
    $("#instructions").hide();
    $("#tutorial").hide();
  });
});

$(function () {
  $('[data-toggle="tooltip"]').tooltip({
    delay: { show: 100, hide: 100 } /* Adjust delay times here */,
  });
});

$(function () {
  $("#join").click(function () {
    socket.emit("join", {});
    $("#join").attr("disabled", true);
    $("#create").attr("disabled", true);
  });
});

$(function () {
  $("#leave").click(function () {
    socket.emit("leave", {});
    $("#leave").attr("disabled", true);
  });
});

// Human guidance input handlers
$(function () {
  $("#sendHumanMessage").click(function () {
    sendHumanMessage();
  });

  // Auto-grow textarea so the full message stays visible
  function autoGrowGuidanceInput() {
    const el = document.getElementById("humanMessageInput");
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  // Initialize height + update on input
  $("#humanMessageInput").on("input", function () {
    autoGrowGuidanceInput();
  });
  autoGrowGuidanceInput();

  // Enter sends; Shift+Enter inserts newline
  $("#humanMessageInput").on("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendHumanMessage();
      return false;
    }
  });
});

function sendHumanMessage() {
  const message = $("#humanMessageInput").val().trim();
  if (!message) {
    return;
  }
  const type = $("input[name='guidanceType']:checked").val() || "STEP";
  const prefixed =
    message.toLowerCase().startsWith("step:") ||
    message.toLowerCase().startsWith("pref:")
      ? message
      : `${type}: ${message}`;
  socket.emit("human_message", { message: prefixed });
  $("#humanMessageInput").val("");
  // Reset textarea height after sending
  const el = document.getElementById("humanMessageInput");
  if (el) {
    el.style.height = "auto";
  }
}

/* * * * * * * * * * * * *
 * Socket event handlers *
 * * * * * * * * * * * * */

window.intervalID = -1;
window.spectating = true;

socket.on("waiting", function (data) {
  // Show game lobby
  $("#error-exit").hide();
  $("#waiting").hide();
  $("#game-over").hide();
  $("#instructions").hide();
  $("#tutorial").hide();
  $("#overcooked").empty();
  $("#lobby").show();
  $("#join").hide();
  $("#join").attr("disabled", true);
  $("#create").hide();
  $("#create").attr("disabled", true);
  $("#leave").show();
  $("#leave").attr("disabled", false);
  if (!data.in_game) {
    // Begin pinging to join if not currently in a game
    if (window.intervalID === -1) {
      window.intervalID = setInterval(function () {
        socket.emit("join", {});
      }, 1000);
    }
  }
});

socket.on("creation_failed", function (data) {
  // Tell user what went wrong
  let err = data["error"];
  $("#overcooked").empty();
  $("#lobby").hide();
  $("#instructions").show();
  $("#tutorial").show();
  $("#waiting").hide();
  $("#join").show();
  $("#join").attr("disabled", false);
  $("#create").show();
  $("#create").attr("disabled", false);
  $("#overcooked").append(
    `<h4>Sorry, game creation code failed with error: ${JSON.stringify(err)}</>`
  );
});

socket.on("start_game", function (data) {
  // Hide game-over and lobby, show game title header
  if (window.intervalID !== -1) {
    clearInterval(window.intervalID);
    window.intervalID = -1;
  }
  graphics_config = {
    container_id: "overcooked",
    start_info: data.start_info,
  };
  window.spectating = data.spectating;
  window.llm_present = data.llm_present;
  $("#error-exit").hide();
  $("#overcooked").empty();
  $("#game-over").hide();
  $("#lobby").hide();
  $("#waiting").hide();
  $("#join").hide();
  $("#join").attr("disabled", true);
  $("#create").hide();
  $("#create").attr("disabled", true);
  $("#instructions").hide();
  $("#tutorial").hide();
  $("#leave").show();
  $("#leave").attr("disabled", false);
  $("#game-title").show();

  if (window.llm_present) {
    $("#human-guidance").show();
  } else {
    $("#human-guidance").hide();
  }

  if (!window.spectating) {
    enable_key_listener();
  }

  graphics_start(graphics_config);
});

socket.on("reset_game", function (data) {
  graphics_end();
  if (!window.spectating) {
    disable_key_listener();
  }

  $("#overcooked").empty();
  $("#reset-game").show();
  setTimeout(function () {
    $("reset-game").hide();
    graphics_config = {
      container_id: "overcooked",
      start_info: data.state,
    };
    if (!window.spectating) {
      enable_key_listener();
    }
    graphics_start(graphics_config);
  }, data.timeout);
});

socket.on("state_pong", function (data) {
  // Draw state update
  drawState(data["state"]);
});

socket.on("end_game", function (data) {
  // Hide game data and display game-over html
  graphics_end();
  if (!window.spectating) {
    disable_key_listener();
  }
  $("#human-guidance").hide();
  $("#game-title").hide();
  $("#game-over").show();
  $("#join").show();
  $("#join").attr("disabled", false);
  $("#create").show();
  $("#create").attr("disabled", false);
  $("#instructions").show();
  $("#tutorial").show();
  $("#leave").hide();
  $("#leave").attr("disabled", true);

  // Game ended unexpectedly
  if (data.status === "inactive") {
    $("#error-exit").show();
  }
});

socket.on("end_lobby", function () {
  // Hide lobby
  $("#lobby").hide();
  $("#human-guidance").hide();
  $("#join").show();
  $("#join").attr("disabled", false);
  $("#create").show();
  $("#create").attr("disabled", false);
  $("#leave").hide();
  $("#leave").attr("disabled", true);
  $("#instructions").show();
  $("#tutorial").show();

  // Stop trying to join
  clearInterval(window.intervalID);
  window.intervalID = -1;
});

/* * * * * * * * * * * * * * * * * * *
 * Confirmation Button Event Handlers *
 * * * * * * * * * * * * * * * * * * */

// Receive confirmation_required event
socket.on("confirmation_required", function (data) {
  // Update button text
  $("#confirmation-text").text(data.display_text);

  // Show button
  const container = $("#confirmation-container");
  container.show();

  // Store data for confirmation
  container.data("actionType", data.action_type);
  container.data("ingredientName", data.ingredient_name);
});

// Button click handler
$(function () {
  $("#confirmation-button").click(function (e) {
    // Prevent any form submit / page refresh
    if (e && typeof e.preventDefault === "function") e.preventDefault();
    const container = $("#confirmation-container");

    // Disable button to prevent double-click
    $(this).prop("disabled", true);

    // Send confirmation to server
    socket.emit("confirm_action", {
      action_type: container.data("actionType"),
      ingredient_name: container.data("ingredientName"),
    });

    // Hide immediately (will be confirmed by button_dismissed)
    container.hide();
    $(this).prop("disabled", false);
  });
});

// Receive button_dismissed event
socket.on("button_dismissed", function () {
  const container = $("#confirmation-container");
  container.hide();
  $("#confirmation-button").prop("disabled", false);
});

/* * * * * * * * * * * * * *
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
  $(document).on("keydown", function (e) {
    // Ignore key events when typing in form controls
    const tag = e.target.tagName.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) {
      return;
    }

    let action = "STAY";
    switch (e.which) {
      case 37: // left
        action = "LEFT";
        break;

      case 38: // up
        action = "UP";
        break;

      case 39: // right
        action = "RIGHT";
        break;

      case 40: // down
        action = "DOWN";
        break;

      case 32: //space
        action = "SPACE";
        break;

      default: // exit this handler for other keys
        return;
    }
    e.preventDefault();
    socket.emit("action", { action: action });
  });
}

function disable_key_listener() {
  $(document).off("keydown");
}

/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function (arr) {
  let retval = {};
  for (let i = 0; i < arr.length; i++) {
    elem = arr[i];
    key = elem["name"];
    value = elem["value"];
    retval[key] = value;
  }
  return retval;
};
