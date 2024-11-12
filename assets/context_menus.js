
if (!window.dash_clientside) { window.dash_clientside = {}; }

window.dash_clientside.clientside = {
    get_text_content: function(n_clicks, content_div_id) {
        // console.log("get_text_content: ", n_clicks, content_div_id);
        if (n_clicks > 0) {
            var contentDiv = document.getElementById(content_div_id);

            // console.log("get_text_content: returns ", contentDiv.textContent);
            return contentDiv.textContent;  // Return the text content of the div
        }
        return '';  // Return empty string initially or when no clicks
    }
}



function add_context_menu_to_neuron_representation() {
    var targets = document.querySelectorAll('.layer-representation');
    if (targets.length > 0) {
        console.log('Adding context menu to neuron representation: ', targets);
        targets.forEach(target => {
            target.oncontextmenu = function(event) {
                event.preventDefault();  // Prevent the default context menu
                event.stopPropagation();
                close_all_context_menus();
                var neuronCtxMenu = document.getElementById(
                    "neuron-context-menu");
                var lastRightClicked = document.getElementById(
                    "last-right-clicked-id");
                lastRightClicked.textContent = event.target.id;
                // lastRightClicked.children = event.target.id;
                window.dash_clientside.clientside.lastRightClickedId = event.target.id
                neuronCtxMenu.style.display = "block";
                neuronCtxMenu.style.left = (event.pageX - 10)+"px";
                neuronCtxMenu.style.top = (event.pageY - 10)+"px";

                // // Remove the menu on click
                // neuronCtxMenu.onclick = close_all_context_menus();
                return false;  // Prevent default action
            };
        });

        
        return true;
    } else {
        return false;
    }
}

function add_context_menu_to_layer_representation() {
    // var targets = document.querySelectorAll('.layer-representation-');
    var targets = document.querySelectorAll('[id*="layer-representation-"]');
    if (targets.length > 0) {
        // console.log('Adding context menu to layer representation: ', targets);
        targets.forEach(target => {
            target.oncontextmenu = function(event) {
                event.preventDefault();  // Prevent the default context menu
                // var neuronCtxMenu = document.getElementById(
                //     "neuron-context-menu");
                // if (neuronCtxMenu.style.display == "block") {
                //     return false;
                // }
                close_all_context_menus();
                // close_all_context_menus();
                var layerCtxMenu = document.getElementById(
                    "layer-context-menu");
                var lastRightClicked = document.getElementById(
                    "last-right-clicked-id");

                lastRightClicked.textContent = event.target.id;              
                layerCtxMenu.style.display = "block";
                layerCtxMenu.style.left = (event.pageX - 10)+"px";
                layerCtxMenu.style.top = (event.pageY - 10)+"px";

                // // Remove the menu on click
                // layerCtxMenu.onclick = close_all_context_menus();
                return false;  // Prevent default action
            };
        });

        
        return true;
    } else {
        return false;
    }
}

function add_context_menu_to_graphs() {
    var targets = document.querySelectorAll('[id*="graphs"]');
    if (targets.length > 0) {
        console.log('Adding context menu to graphs: ', targets);
        targets.forEach(target => {
            target.oncontextmenu = function(event) {
                event.preventDefault();  // Prevent the default context menu
                close_all_context_menus();
                var graphCtxMenu = document.getElementById(
                    "plots-context-menu");
                var lastRightClicked = document.getElementById(
                    "last-right-clicked-id");
                lastRightClicked.textContent = event.target.id;
               
                graphCtxMenu.style.display = "block";
                graphCtxMenu.style.left = (event.pageX - 10)+"px";
                graphCtxMenu.style.top = (event.pageY - 10)+"px";
                // // Remove the menu on click
                // graphCtxMenu.onclick = close_all_context_menus();
                return false;  // Prevent default action
            };
        });
        return true;
    } else {
        return false;
    }
}


function close_all_context_menus() {
    const menus = document.querySelectorAll('[id*="menu"]');
    console.log("close all context menus", menus);
    menus.forEach(menu => {
        menu.style.display = '';
        menu.style.left = '';
        menu.style.top = '';
    });
}

function checkAndExecute() {
    var added_callback = add_context_menu_to_neuron_representation();
    added_callback = added_callback && add_context_menu_to_graphs();
    added_callback = added_callback && add_context_menu_to_layer_representation()
    document.onclick = close_all_context_menus;

    if (!added_callback) {
        setTimeout(checkAndExecute, 500); // Check every 500 ms
    }
}
// document.addEventListener('DOMContentLoaded', checkAndExecute);

