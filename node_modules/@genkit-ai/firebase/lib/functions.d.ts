import { FlowWrapper, FlowAuthPolicy, StepsFunction } from '@genkit-ai/flow';
import * as express from 'express';
import { HttpsFunction, HttpsOptions } from 'firebase-functions/v2/https';
import * as z from 'zod';

/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

type FunctionFlow<I extends z.ZodTypeAny, O extends z.ZodTypeAny, S extends z.ZodTypeAny> = HttpsFunction & FlowWrapper<I, O, S>;
interface FunctionFlowAuth<I extends z.ZodTypeAny> {
    provider: express.RequestHandler;
    policy: FlowAuthPolicy<I>;
}
interface FunctionFlowConfig<I extends z.ZodTypeAny, O extends z.ZodTypeAny, S extends z.ZodTypeAny> {
    name: string;
    inputSchema?: I;
    outputSchema?: O;
    authPolicy: FunctionFlowAuth<I>;
    streamSchema?: S;
    httpsOptions?: HttpsOptions;
    enforceAppCheck?: boolean;
    consumeAppCheckToken?: boolean;
}
/**
 * Creates a flow backed by Cloud Functions for Firebase gen2 HTTPS function.
 */
declare function onFlow<I extends z.ZodTypeAny, O extends z.ZodTypeAny, S extends z.ZodTypeAny>(config: FunctionFlowConfig<I, O, S>, steps: StepsFunction<I, O, S>): FunctionFlow<I, O, S>;
/**
 * Indicates that no authorization is in effect.
 *
 * WARNING: If you are using Cloud Functions for Firebase with no IAM policy,
 * this will allow anyone on the internet to execute this flow.
 */
declare function noAuth(): FunctionFlowAuth<any>;

export { type FunctionFlow, type FunctionFlowAuth, noAuth, onFlow };
