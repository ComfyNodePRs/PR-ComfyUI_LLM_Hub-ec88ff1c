import {app} from "../../scripts/app.js";
import {createTextWidget} from "./utils.js"

app.registerExtension({
    name: "LLM_Hub.Generated_Output",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Generated_Output") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                const text = createTextWidget(app, this, "text");
                const nodeWidth = this.size[0];
                const nodeHeight = this.size[1];
                this.setSize([nodeWidth, nodeHeight * 3]);
                return result;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                this.widgets.find(obj => obj.name === "text").value = message.text;
            };
        }
    },
});