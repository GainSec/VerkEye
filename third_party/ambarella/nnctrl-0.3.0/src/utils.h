/*******************************************************************************
 * utils.h
 *
 * History:
 *    2018/08/22  - [Tao Wu] created
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	   See the NOTICE file
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

#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdint.h>
#include <pthread.h>

#define ALIGN_64	(64)
#define ALIGN_32	(32)

#ifndef ROUND_UP
#define ROUND_UP(size, align) (((size) + ((align) - 1)) & ~((align) - 1))
#endif

#ifndef MINUS_ONE
#define MINUS_ONE(size) ((size) -1)
#endif

static inline void API_LOCK(pthread_mutex_t *lock)
{
	if (pthread_mutex_lock(lock) < 0) {
		perror("mutex_lock");
	}
}

static inline void API_UNLOCK(pthread_mutex_t *lock)
{
	if (pthread_mutex_unlock(lock) < 0) {
		perror("mutex_unlock");
	}
}

uint32_t layout_mem(struct net_desc *pnet, uint32_t size);
int load_file(struct net_desc *pnet, const char *file_name, uint32_t offset, uint32_t size);
int store_file(struct net_desc *pnet, const char *file_name, uint32_t offset, uint32_t size);

struct net_desc *get_net_desc(struct nnctrl_info *pctl, int net_id);
uint64_t get_audio_clk(int fd_cav);

#endif
