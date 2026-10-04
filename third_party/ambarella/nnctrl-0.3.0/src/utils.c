/*******************************************************************************
 * utils.c
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include "nnctrl_priv.h"
#include "utils.h"

uint32_t layout_mem(struct net_desc *pnet, uint32_t size)
{
	uint32_t pos = pnet->mem_offset;

	if (pnet->verbose) {
		printf("--> layout_mem of size [%u], mem_offset [%u]\n",
			size, pnet->mem_offset);
	}
	/* Use 64 btye alignment to improve 64bit DRAM access efficiency */
	pnet->mem_offset += ROUND_UP(size, ALIGN_64);

	return pos;
}

int load_file(struct net_desc *pnet, const char *file_name, uint32_t offset, uint32_t size)
{
	FILE *in = NULL;
	uint32_t file_size = 0;
	int rval = 0;

	do {
		in = fopen(file_name, "rb");
		if (in == NULL) {
			perror(file_name);
			rval = -1;
			break;
		}
		if (fseek(in, 0, SEEK_END) < 0) {
			perror("fseek input file");
			rval = -1;
			break;
		}
		file_size = ftell(in);
		if (file_size != size) {
			printf("blob size [%u] doesn't match input file size [%u]\n",
				size, file_size);
			rval = -1;
			break;
		}
		if (fseek(in, 0, SEEK_SET) < 0) {
			perror("fseek input file");
			rval = -1;
			break;
		}
		if (fread(pnet->virt_addr + offset, 1, size, in) != size) {
			perror("fread input file");
			rval = -1;
			break;
		}
	} while (0);

	if (in) {
		fclose(in);
		in = NULL;
	}

	return rval;
}

int store_file(struct net_desc *pnet, const char *file_name, uint32_t offset, uint32_t size)
{
	FILE *out = NULL;
	int rval = 0;

	do {
		if ((out = fopen(file_name, "w+b")) == NULL) {
			perror(file_name);
			rval = -1;
			break;
		}
		if (fwrite(pnet->virt_addr + offset, 1, size, out) != size) {
			perror(file_name);
			rval = -1;
			break;
		}
	} while (0);

	if (out) {
		fclose(out);
		out = NULL;
	}

	return rval;
}

struct net_desc *get_net_desc(struct nnctrl_info *pctl, int net_id)
{
	struct net_desc *pnet = NULL, *safe = NULL;
	uint32_t found = 0;

	if (!pctl->init_done) {
		printf("nnctrl library is not inited\n");
		return NULL;
	}

	API_LOCK(&pctl->list_lock);
	if (!list_empty(&pctl->net_list)) {
		list_for_each_entry_safe(pnet, safe, &pctl->net_list, node) {
			if (pnet->net_id == net_id) {
				found = 1;
				break;
			}
		}
	}
	API_UNLOCK(&pctl->list_lock);
	if (!found) {
		printf("Not found net_id: %d\n", net_id);
		return NULL;
	}
	if (!pnet->net_init_done) {
		printf("Network id: %d is not inited\n", net_id);
		return NULL;
	}

	return pnet;
}

uint64_t get_audio_clk(int fd_cav)
{
	uint64_t value = 0;

	if (ioctl(fd_cav, CAVALRY_GET_AUDIO_CLK, &value) < 0) {
		perror("CAVALRY_GET_AUDIO_CLK");
		value = DEFAULT_AUDIO_CLK_HZ;
	}

	return value;
}
