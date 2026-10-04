/*******************************************************************************
 * mem_ver.h
 *
 * History:
 *    2018/09/18  - [Tao Wu] created
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
******************************************************************************/

#ifndef _MEM_VER_H_
#define _MEM_VER_H_

#define MEM_LIB_MAJOR 0
#define MEM_LIB_MINOR 0
#define MEM_LIB_PATCH 6
#define MEM_LIB_VERSION ((MEM_LIB_MAJOR << 16) | \
			(MEM_LIB_MINOR << 8)  | \
			MEM_LIB_PATCH)

#endif
