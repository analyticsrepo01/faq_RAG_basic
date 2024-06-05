import { StepsFunction } from '@genkit-ai/flow';
import { TaskQueueOptions } from 'firebase-functions/v2/tasks';
import * as z from 'zod';
import { FunctionFlow } from './functions.mjs';
import 'express';
import 'firebase-functions/v2/https';

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

interface ScheduledFlowConfig<I extends z.ZodTypeAny = z.ZodTypeAny, O extends z.ZodTypeAny = z.ZodTypeAny, S extends z.ZodTypeAny = z.ZodTypeAny> {
    name: string;
    inputSchema?: I;
    outputSchema?: O;
    streamSchema?: S;
    taskQueueOptions?: TaskQueueOptions;
}
/**
 * Creates a scheduled flow backed by Cloud Functions for Firebase gen2 Cloud Task triggered function.
 * This feature is EXPERIMENTAL -- APIs will change or may get removed completely.
 * For testing and feedback only.
 */
declare function onScheduledFlow<I extends z.ZodTypeAny = z.ZodTypeAny, O extends z.ZodTypeAny = z.ZodTypeAny, S extends z.ZodTypeAny = z.ZodTypeAny>(config: ScheduledFlowConfig<I, O, S>, steps: StepsFunction<I, O, S>): FunctionFlow<I, O, S>;

export { onScheduledFlow };
