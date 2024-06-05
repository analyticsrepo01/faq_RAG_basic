import {
  __async,
  __spreadProps,
  __spreadValues
} from "./chunk-DJRN6NKF.mjs";
import { OperationSchema } from "@genkit-ai/core";
import { logger } from "@genkit-ai/core/logging";
import {
  defineFlow
} from "@genkit-ai/flow";
import { getAppCheck } from "firebase-admin/app-check";
import {
  onRequest
} from "firebase-functions/v2/https";
import {
  callHttpsFunction,
  getLocation,
  initializeAppIfNecessary
} from "./helpers.js";
function onFlow(config, steps) {
  const f = defineFlow(
    __spreadProps(__spreadValues({}, config), {
      authPolicy: config.authPolicy.policy,
      invoker: (flow, data, streamingCallback) => __async(this, null, function* () {
        const responseJson = yield callHttpsFunction(
          flow.name,
          yield getLocation(),
          data,
          streamingCallback
        );
        const res = JSON.parse(responseJson);
        if (streamingCallback) {
          return OperationSchema.parse(res);
        } else {
          return {
            name: "",
            done: true,
            result: {
              response: res
            }
          };
        }
      })
    }),
    steps
  );
  const wrapped = wrapHttpsFlow(f, config);
  const funcFlow = wrapped;
  funcFlow.flow = f;
  return funcFlow;
}
function wrapHttpsFlow(flow, config) {
  var _a;
  return onRequest(
    __spreadProps(__spreadValues({}, config.httpsOptions), {
      memory: ((_a = config.httpsOptions) == null ? void 0 : _a.memory) || "512MiB"
    }),
    (req, res) => __async(this, null, function* () {
      if (config.enforceAppCheck) {
        if (!(yield appCheckValid(
          req.headers["x-firebase-appcheck"],
          !!config.consumeAppCheckToken
        ))) {
          const respBody = {
            error: {
              status: "UNAUTHENTICATED",
              message: "Unauthorized"
            }
          };
          logger.logStructured(`Response[/${flow.name}]`, {
            path: `/${flow.name}`,
            code: 401,
            body: respBody
          });
          res.status(401).send(respBody).end();
          return;
        }
      }
      yield config.authPolicy.provider(
        req,
        res,
        () => flow.expressHandler(req, res)
      );
    })
  );
}
function appCheckValid(tok, consume) {
  return __async(this, null, function* () {
    if (typeof tok !== "string")
      return false;
    initializeAppIfNecessary();
    try {
      const appCheckClaims = yield getAppCheck().verifyToken(tok, { consume });
      if (consume && appCheckClaims.alreadyConsumed)
        return false;
      return true;
    } catch (e) {
      return false;
    }
  });
}
function noAuth() {
  return {
    provider: (req, res, next) => next(),
    policy: () => {
    }
  };
}
export {
  noAuth,
  onFlow
};
//# sourceMappingURL=functions.mjs.map