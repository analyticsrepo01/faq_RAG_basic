import {
  __async,
  __spreadProps,
  __spreadValues
} from "./chunk-DJRN6NKF.mjs";
import { OperationSchema } from "@genkit-ai/core";
import { durableFlow } from "@genkit-ai/flow/experimental";
import { getFunctions } from "firebase-admin/functions";
import { logger } from "firebase-functions/v2";
import {
  onTaskDispatched
} from "firebase-functions/v2/tasks";
import { callHttpsFunction, getFunctionUrl, getLocation } from "./helpers.js";
function onScheduledFlow(config, steps) {
  const f = durableFlow(
    __spreadProps(__spreadValues({}, config), {
      invoker: (flow, data, streamingCallback) => __async(this, null, function* () {
        const responseJson = yield callHttpsFunction(
          flow.name,
          yield getLocation(),
          data,
          streamingCallback
        );
        return OperationSchema.parse(JSON.parse(responseJson));
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
  const tq = onTaskDispatched(
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
    const queue = getFunctions().taskQueue(flowName);
    const targetUri = yield getFunctionUrl(flowName, "us-central1");
    logger.debug(
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
export {
  onScheduledFlow
};
//# sourceMappingURL=experimental.mjs.map