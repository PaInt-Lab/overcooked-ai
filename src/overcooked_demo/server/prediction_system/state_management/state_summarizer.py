"""Summarizes game state into predicates for decision making."""

from typing import Dict


class StateSummarizer:
    """Summarizes game state into high-level predicates."""
    
    def __init__(self):
        # Processing state tracking
        self.onion_chopped = False
        self.tomato_chopped = False
        self.onion_washed = False
        self.tomato_washed = False
        self.onion_salted = False
        self.onion_peppered = False
        self.tomato_salted = False
        self.tomato_peppered = False
        
        # References to tile manager (set externally)
        self.tile_manager = None
    
    def summarize_state(self, state, agent_index: int, info: dict = None) -> Dict:
        """Extract state predicates for the LLM."""
        if info is None:
            info = {}
            
        sd = state.to_dict()

        def held_item(player_dict):
            held = player_dict["held_object"]
            if held is None:
                return "none"
            if held.get("name") in ("onion", "tomato"):
                return held["name"]
            if held.get("ingredient") in ("onion", "tomato"):
                return held["ingredient"]
            if held.get("name") in ("dish", "soup"):
                return held["name"]
            return "none"

        me = sd["players"][agent_index]
        them = sd["players"][1 - (agent_index or 0)]
        agent_item = held_item(me)
        partner_item = held_item(them)

        onion_hand = "none"
        if agent_item == "onion":
            onion_hand = "agent"
        elif partner_item == "onion":
            onion_hand = "partner"

        tomato_hand = "none"
        if agent_item == "tomato":
            tomato_hand = "agent"
        elif partner_item == "tomato":
            tomato_hand = "partner"

        dish_hand = "none"
        if agent_item == "dish":
            dish_hand = "agent"
        elif partner_item == "dish":
            dish_hand = "partner"

        soup_hand = "none"
        if agent_item == "soup":
            soup_hand = "agent"
        elif partner_item == "soup":
            soup_hand = "partner"

        tile_contents = {}
        onion_in_pot = False
        tomato_in_pot = False
        soup_cooking = False
        soup_ready = False
        soup_in_pot_not_cooking = False
        
        for obj in sd["objects"]:
            p = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
            
            if obj.get("name") == "soup" and obj.get("_ingredients"):
                for ingredient in obj["_ingredients"]:
                    ing_name = ingredient.get("name")
                    if ing_name:
                        tile_contents.setdefault(p, []).append(ing_name)
                
                if self.tile_manager and p in self.tile_manager.stove_tiles:
                    for ingredient in obj["_ingredients"]:
                        ing_name = ingredient.get("name")
                        if ing_name in ("onion"):
                            onion_in_pot = True
                        elif ing_name in ("tomato"):
                            tomato_in_pot = True
                    
                    cooking_tick = obj.get("cooking_tick", -1)
                    is_cooking = obj.get("is_cooking", False)
                    is_ready = obj.get("is_ready", False)
                    
                    if is_cooking and cooking_tick >= 1:
                        soup_cooking = True
                    elif is_ready:
                        soup_ready = True
                    elif cooking_tick == -1:
                        soup_in_pot_not_cooking = True

        # Check staging and processing stations
        onion_staged = False
        onion_at_chopping = False
        tomato_staged = False
        tomato_at_chopping = False
        dish_staged = False
        soup_staged = False
        onion_at_sink = False
        tomato_at_sink = False
        onion_at_salt_station = False
        tomato_at_salt_station = False
        onion_at_pepper_station = False
        tomato_at_pepper_station = False
        
        if self.tile_manager:
            onion_staged = any(
                "onion" in tile_contents.get(pos, [])
                for pos in self.tile_manager.onion_staging_tiles
            )
            
            onion_at_chopping = any(
                "onion" in tile_contents.get(pos, [])
                for pos in self.tile_manager.onion_chopping_stations
            )

            # MODIFIED: Only mark as chopped if object state is actually 'chopped'
            # This requires human confirmation button click before processing
            if onion_at_chopping:
                # Check if the onion at the chopping station is actually chopped
                for pos in self.tile_manager.onion_chopping_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'onion' and hasattr(obj, 'state') and obj.state == 'chopped':
                        self.onion_chopped = True
                        break

            tomato_staged = any(
                "tomato" in tile_contents.get(pos, [])
                for pos in self.tile_manager.tomato_staging_tiles
            )
            
            tomato_at_chopping = any(
                "tomato" in tile_contents.get(pos, [])
                for pos in self.tile_manager.tomato_chopping_stations
            )

            # MODIFIED: Only mark as chopped if object state is actually 'chopped'
            if tomato_at_chopping:
                for pos in self.tile_manager.tomato_chopping_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'tomato' and hasattr(obj, 'state') and obj.state == 'chopped':
                        self.tomato_chopped = True
                        break

            dish_staged = any(
                "dish" in tile_contents.get(pos, [])
                for pos in self.tile_manager.dish_staging_tiles
            )

            soup_staged = any(
                "soup" in tile_contents.get(pos, [])
                for pos in self.tile_manager.soup_staging_tiles
            )

            # Check sink-related states
            onion_at_sink = any(
                "onion" in tile_contents.get(pos, [])
                for pos in self.tile_manager.sink_stations
            )
            
            tomato_at_sink = any(
                "tomato" in tile_contents.get(pos, [])
                for pos in self.tile_manager.sink_stations
            )
            
            # MODIFIED: Track washing completion state - check actual object state
            if onion_at_sink:
                for pos in self.tile_manager.sink_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'onion' and hasattr(obj, 'state') and obj.state == 'washed':
                        self.onion_washed = True
                        break

            if tomato_at_sink:
                for pos in self.tile_manager.sink_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'tomato' and hasattr(obj, 'state') and obj.state == 'washed':
                        self.tomato_washed = True
                        break

            # Check salt station-related states
            onion_at_salt_station = any(
                "onion" in tile_contents.get(pos, [])
                for pos in self.tile_manager.salt_stations
            )
            
            tomato_at_salt_station = any(
                "tomato" in tile_contents.get(pos, [])
                for pos in self.tile_manager.salt_stations
            )
            
            # MODIFIED: Track salting completion state - check if object has 'salted' property
            if onion_at_salt_station:
                for pos in self.tile_manager.salt_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'onion':
                        # Check if object has salted property
                        if hasattr(obj, 'properties') and 'salted' in obj.properties:
                            self.onion_salted = True
                            break

            if tomato_at_salt_station:
                for pos in self.tile_manager.salt_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'tomato':
                        if hasattr(obj, 'properties') and 'salted' in obj.properties:
                            self.tomato_salted = True
                            break

            # Check pepper station-related states
            onion_at_pepper_station = any(
                "onion" in tile_contents.get(pos, [])
                for pos in self.tile_manager.pepper_stations
            )
            
            tomato_at_pepper_station = any(
                "tomato" in tile_contents.get(pos, [])
                for pos in self.tile_manager.pepper_stations
            )
            
            # MODIFIED: Track peppering completion state - check if object has 'peppered' property
            if onion_at_pepper_station:
                for pos in self.tile_manager.pepper_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'onion':
                        if hasattr(obj, 'properties') and 'peppered' in obj.properties:
                            self.onion_peppered = True
                            break

            if tomato_at_pepper_station:
                for pos in self.tile_manager.pepper_stations:
                    obj = state.get_object(pos)
                    if obj and obj.name == 'tomato':
                        if hasattr(obj, 'properties') and 'peppered' in obj.properties:
                            self.tomato_peppered = True
                            break

        soup_served = False
        soup_delivered_by = "none"
        
        # Check if soup was served (from event_infos in info dict)
        # This is set by the game environment when soup delivery is detected
        if info and info.get("event_infos", {}).get("soup_delivery"):
            soup_delivery = info.get("event_infos", {}).get("soup_delivery", [False, False])
            if any(soup_delivery):
                soup_served = True
                if soup_delivery[0]:  # Agent delivered
                    soup_delivered_by = "agent"
                elif soup_delivery[1]:  # Partner delivered
                    soup_delivered_by = "partner"

        return {
            "onion_hand": onion_hand,
            "onion_staged": onion_staged,
            "onion_at_chopping": onion_at_chopping,
            "onion_chopped": self.onion_chopped,
            "onion_at_sink": onion_at_sink,
            "onion_washed": self.onion_washed,
            "onion_at_salt_station": onion_at_salt_station,
            "onion_salted": self.onion_salted,
            "onion_at_pepper_station": onion_at_pepper_station,
            "onion_peppered": self.onion_peppered,
            "tomato_hand": tomato_hand,
            "tomato_staged": tomato_staged,
            "tomato_at_chopping": tomato_at_chopping,
            "tomato_chopped": self.tomato_chopped,
            "tomato_at_sink": tomato_at_sink,
            "tomato_washed": self.tomato_washed,
            "tomato_at_salt_station": tomato_at_salt_station,
            "tomato_salted": self.tomato_salted,
            "tomato_at_pepper_station": tomato_at_pepper_station,
            "tomato_peppered": self.tomato_peppered,
            "onion_in_pot": onion_in_pot,
            "tomato_in_pot": tomato_in_pot,
            "soup_cooking": soup_cooking,
            "soup_ready": soup_ready,
            "soup_in_pot_not_cooking": soup_in_pot_not_cooking,
            "dish_hand": dish_hand,
            "dish_staged": dish_staged,
            "soup_hand": soup_hand,
            "soup_staged": soup_staged,
            "soup_served": soup_served,
            "soup_delivered_by": soup_delivered_by
        }
    
    def reset_processing_states(self):
        """Reset processing states (called when soup is served)."""
        self.onion_chopped = False
        self.tomato_chopped = False
        self.onion_washed = False
        self.tomato_washed = False
        self.onion_salted = False
        self.onion_peppered = False
        self.tomato_salted = False
        self.tomato_peppered = False

