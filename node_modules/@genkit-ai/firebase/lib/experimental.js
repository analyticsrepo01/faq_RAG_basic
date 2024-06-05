"use strict";
var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var experimental_exports = {};
__export(experimental_exports, {
  onScheduledFlow: () => onScheduledFlow
});
module.exports = __toCommonJS(experimental_exports);
var import_core = require("@genkit-ai/core");
var import_experimental = require("@genkit-ai/flow/experimental");
var import_functions = require("firebase-admin/functions");
var import_v2 = require("firebase-functions/v2");
var import_tasks = require("firebase-functions/v2/tasks");
var import_helpers = require("./helpers.js");
function onScheduledFlow(config, steps) {
  const f = (0, import_experimental.durableFlow)(
    __spreadProps(__spreadValues({}, config), {
      invoker: (flow, data, streamingCallback) => __async(this, null, function* () {
        const responseJson = yield (0, import_helpers.callHttpsFunction)(
          flow.name,
          yield (0, import_helpers.getLocation)(),
          data,
          streamingCallback
        );
        return import_core.OperationSchema.parse(JSON.parse(responseJson));
      }),
      scheduler: (flow, msg, delaySeconds) => __async(this, null, function* () {
        yield enqueueCloudTask(flow.name, msg, delaySeconds);
      })
    }),
    steps
  );
  const wrapped = wrapScheduledFlow(f, config);
  const funcFlow = wrapped;
  funcFlow.flow = f;
  return funcFlow;
}
function wrapScheduledFlow(flow, config) {
  var _a, _b;
  const tq = (0, import_tasks.onTaskDispatched)(
    __spreadProps(__spreadValues({}, config.taskQueueOptions), {
      memory: ((_a = config.taskQueueOptions) == null ? void 0 : _a.memory) || "512MiB",
      retryConfig: ((_b = config.taskQueueOptions) == null ? void 0 : _b.retryConfig) || {
        maxAttempts: 2,
        minBackoffSeconds: 10
      }
    }),
    () => {
    }
    // never called, everything handled in createControlAPI.
  );
  return createControlAPI(flow, tq);
}
function createControlAPI(flow, tq) {
  const interceptor = flow.expressHandler;
  interceptor.__endpoint = tq.__endpoint;
  if (tq.hasOwnProperty("__requiredAPIs")) {
    interceptor.__requiredAPIs = tq["__requiredAPIs"];
  }
  return interceptor;
}
function enqueueCloudTask(flowName, payload, scheduleDelaySeconds) {
  return __async(this, null, function* () {
    const queue = (0, import_functions.getFunctions)().taskQueue(flowName);
    const targetUri = yield (0, import_helpers.getFunctionUrl)(flowName, "us-central1");
    import_v2.logger.debug(
      `dispatchCloudTask targetUri for flow ${flowName} with delay ${scheduleDelaySeconds}`
    );
    yield queue.enqueue(payload, {
      scheduleDelaySeconds,
      dispatchDeadlineSeconds: scheduleDelaySeconds,
      uri: targetUri,
      headers: {
        "Content-Type": "application/json"
      }
    });
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  onScheduledFlow
});
//# sourceMappingURL=experimental.js.map