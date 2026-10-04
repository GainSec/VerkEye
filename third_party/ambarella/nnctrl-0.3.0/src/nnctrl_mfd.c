/*******************************************************************************
 * nnctrl_mfd.c
 *
 * History:
 *    2020/05/15  - [Tao Wu] created
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/errno.h>
#include <cavalry_ioctl.h>
#include <nnctrl.h>

#include "nnctrl_priv.h"
#include "nnctrl_ver.h"
#include "parser.h"
#include "port_cfg_mfd.h"
#include "rundags_mfd.h"
#include "debug.h"
#include "utils.h"

int nnctrl_init_net_by_mfd(struct net_cfg *net_cf,
	struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out)
{
	struct nnctrl_info *pctl = NULL;
	struct net_desc *pnet = NULL;
	struct net_desc *_pnet = NULL, *safe = NULL;
	int is_duplicated_id = 0, start_id = 0;
	int rval = 0;

	if (net_cf == NULL) {
		printf("Invalid nnctrl init net param\n");
		return -1;
	}
	if ((net_cf->net_file == NULL) && (net_cf->net_feed_virt == NULL)) {
		printf("Both of net_file and net_virt are NULL\n");
		return -1;
	}
	if ((net_cf->net_file) && (net_cf->net_feed_virt)) {
		printf("Both of net_file and net_virt are valid, please select one way\n");
		return -1;
	}
	if (check_io_num_mfd(net_in, net_out) < 0) {
		return -1;
	}

	pctl = get_nnctrl_global_context();
	if (!pctl->init_done) {
		printf("nnctrl library is not inited\n");
		return -1;
	}
	pnet = (struct net_desc *)malloc(sizeof(struct net_desc));
	if (!pnet) {
		perror("malloc");
		return -1;
	}
	memset((void *)pnet, 0, (sizeof(struct net_desc)
		- sizeof(pnet->run_dags) - sizeof(pnet->partial_dags)
		- sizeof(pnet->run_dags_mfd) - sizeof(pnet->partial_dags_mfd)));

	do {
		if (net_cf->net_file) {
			pnet->net_fp = fopen(net_cf->net_file, "rb");
			if (pnet->net_fp == NULL) {
				perror("fopen");
				printf("nnctrl init net open %s err\n", net_cf->net_file);
				rval = -1;
				break;
			}
		} else {
			pnet->net_feed_virt = net_cf->net_feed_virt;
		}

		if (check_cavalry_gen_bin_header(pnet) < 0) {
			printf("cavalry package header check failed!\n");
			rval = -1;
			break;
		}
		pnet->verbose = net_cf->verbose;
		pnet->reuse_mem = net_cf->reuse_mem;
		pnet->print_time = net_cf->print_time;
		pnet->net_loop_cnt = net_cf->net_loop_cnt;
		pnet->use_memfd = 1;
		pnet->virt_addr = NULL;
		pnet->phy_addr = 0;
		pnet->mem_size = 0;
		pnet->mem_fd = 0;

		if (pnet->verbose) {
			printf("Cavalry Net [%s], total dvi num [%u]\n",
				net_cf->net_file, pnet->header.dvi_num);
		}
		if (pnet->net_loop_cnt > 1) {
			printf("Net Loop Count: %u\n", pnet->net_loop_cnt);
		}

		if (parse_dvi_desc(pnet) < 0) {
			printf("parse dvi descriptor failed!\n");
			rval = -1;
			break;
		}

		if (gen_net_sub_parent_port(pnet) < 0) {
			printf("gen network input and output err\n");
			rval = -1;
			break;
		}

		if (get_port_cfg_mfd(pnet, net_in, net_out) < 0) {
			printf("get port cfg failed!\n");
			rval = -1;
			break;
		}

		if (allocate_port_mem(pnet) < 0) {
			printf("allocate port memory failed!\n");
			rval = -1;
			break;
		}

		gen_net_parent_port_addr(pnet);

		if ((pnet->dvi_mem_align_total + pnet->blob_mem_align_total) != (pnet->mem_offset)) {
			printf("alloc net memory mismatch, %u + %u != %u\n",
				pnet->dvi_mem_align_total, pnet->blob_mem_align_total, pnet->mem_offset);
			rval = -1;
			break;
		}

		if (pthread_mutex_init(&pnet->net_lock, NULL) < 0) {
			perror("nnctrl net lock init");
			rval = -1;
			break;
		}
	} while (0);

	if (rval == 0) {
		INIT_LIST_HEAD(&pnet->node);
		API_LOCK(&pctl->list_lock);
		/* Make sure generate unique net_id */
		start_id = pctl->candidate_id;
		do {
			is_duplicated_id = 0;
			if (!list_empty(&pctl->net_list)) {
				list_for_each_entry_safe(_pnet, safe, &pctl->net_list, node) {
					if (_pnet->net_id == pctl->candidate_id) {
						is_duplicated_id = 1;
						break;
					}
				}
			}
			if (is_duplicated_id) {
				pctl->candidate_id++;
				if (pctl->candidate_id == S32_VALUE_MAX) {
					pctl->candidate_id = 0;
				}
				if (pctl->candidate_id == start_id) {
					printf("No valid Net_id can be used for int32 is full\n");
					rval = -1;
					break;
				}
			}
		} while (is_duplicated_id);
		if (rval == 0) {
			pnet->net_id = pctl->candidate_id;
			pctl->candidate_id++;
			if (pctl->candidate_id == S32_VALUE_MAX) {
				pctl->candidate_id = 0;
			}
			list_add_tail(&pnet->node, &pctl->net_list);
		}
		API_UNLOCK(&pctl->list_lock);
	}

	if (rval < 0) {
		if (pnet->net_fp) {
			fclose(pnet->net_fp);
			pnet->net_fp = NULL;
		} else {
			pnet->net_feed_virt_pos = 0;
			pnet->net_feed_virt = NULL;
		}
		if (pnet) {
			free(pnet);
			pnet = NULL;
		}
	}

	if (rval == 0) {
		net_cf->dvi_mem_total = pnet->dvi_mem_total;
		net_cf->net_mem_total = pnet->dvi_mem_align_total + pnet->blob_mem_align_total;
		net_cf->blob_mem_total = pnet->blob_mem_total;
		net_cf->bandwidth_total = pnet->dvi_mem_total + pnet->blob_bw_total;
		pnet->net_init_done = 1;
		rval = pnet->net_id;  /* Return net id */
	}

	return rval;
}

int nnctrl_load_net_by_mfd(int net_id, struct cavalry_mfd_desc *net_m)
{
	struct nnctrl_info *pctl = NULL;
	struct net_desc *pnet = NULL;
	int rval = 0;

	if (net_m == NULL) {
		printf("Invalid nnctrl load net param\n");
		return -1;
	}
	if ((net_m->virt_addr == NULL) || (net_m->mem_size == 0)) {
		printf("Invalid nnctrl net memory param %p, %lu\n",
			net_m->virt_addr, net_m->mem_size);
		return -1;
	}

	pctl = get_nnctrl_global_context();
	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	pnet->virt_addr = net_m->virt_addr;
	pnet->mem_size = net_m->mem_size;
	pnet->mem_fd = net_m->fd;

	do {
		if (load_dvi_image_bin(pnet) < 0) {
			printf("Load dvi image err\n");
			rval = -1;
			break;
		}

		set_sub_and_parent_port_mfd(pnet);
	} while (0);

	if (pnet->net_fp) {
		fclose(pnet->net_fp);
		pnet->net_fp = NULL;
	} else {
		pnet->net_feed_virt_pos = 0;
		pnet->net_feed_virt = NULL;
	}

	if (pnet->verbose) {
		show_dvi_node_list(pnet);
	}

	if (set_run_dags_struct_mfd(pnet) < 0) {
		printf("init run dags ioctl struct err\n");
		rval = -1;
	}

	if (rval == 0) {
		pnet->net_load_done = 1;
	}

	return rval;
}

int nnctrl_get_net_io_cfg_by_mfd(int net_id,
	struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out)
{
	struct nnctrl_info *pctl = NULL;
	struct net_desc *pnet = NULL;
	uint32_t i = 0, is_loaded = 0;

	if ((net_in == NULL) && (net_out == NULL)) {
		printf("Both of input and output are NULL\n");
		return -1;
	}

	pctl = get_nnctrl_global_context();
	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	is_loaded = pnet->net_init_done;

	API_LOCK(&pnet->net_lock);
	if (net_in) {
		net_in->in_num = pnet->net_in_num;
		for (i = 0; i < pnet->net_in_num; i++) {
			net_in->in_desc[i].name = pnet->net_prt_in[i].name;
			net_in->in_desc[i].size = pnet->net_prt_in[i].port_size;
			net_in->in_desc[i].dim = pnet->net_prt_in[i].dim;
			net_in->in_desc[i].data_fmt = pnet->net_prt_in[i].data_fmt;
			if (is_loaded) { /* Return io addr after nnctrl_load_net() */
				net_in->in_desc[i].mem_fd = pnet->phy_addr + pnet->net_prt_in[i].port_dram_fd;
				net_in->in_desc[i].fd_offset = pnet->phy_addr + pnet->net_prt_in[i].port_dram_addr;
				net_in->in_desc[i].virt = pnet->virt_addr + pnet->net_prt_in[i].port_dram_addr;
			} else {
				net_in->in_desc[i].mem_fd = -1;
				net_in->in_desc[i].fd_offset = 0;
				net_in->in_desc[i].virt = NULL;
			}
		}
	}

	if (net_out) {
		net_out->out_num = pnet->net_out_num;
		for (i = 0; i < pnet->net_out_num; i++) {
			net_out->out_desc[i].name = pnet->net_prt_out[i].name;
			net_out->out_desc[i].size = pnet->net_prt_out[i].port_size;
			net_out->out_desc[i].dim = pnet->net_prt_out[i].dim;
			net_out->out_desc[i].data_fmt = pnet->net_prt_out[i].data_fmt;
			if (is_loaded) { /* Return io addr after nnctrl_load_net() */
				net_out->out_desc[i].mem_fd = pnet->net_prt_out[i].port_dram_fd;
				net_out->out_desc[i].fd_offset = pnet->net_prt_out[i].port_dram_addr;
				net_out->out_desc[i].virt = pnet->virt_addr + pnet->net_prt_out[i].port_dram_addr;
			} else {
				net_out->out_desc[i].mem_fd = -1;
				net_out->out_desc[i].fd_offset = 0;
				net_out->out_desc[i].virt = NULL;
			}
		}
	}
	API_UNLOCK(&pnet->net_lock);

	return 0;
}

int nnctrl_set_net_io_cfg_by_mfd(int net_id,
	struct net_input_mfd_cfg *net_in, struct net_output_mfd_cfg *net_out)
{
	struct nnctrl_info *pctl = NULL;
	struct net_desc *pnet = NULL;
	uint32_t need_update_port = 0, need_update_poke = 0;

	if ((net_in == NULL) && (net_out == NULL)) {
		printf("Both of input and output are NULL\n");
		return -1;
	}

	pctl = get_nnctrl_global_context();
	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	if (!pnet->net_load_done) {
		printf("Network id: %d is not load when set cfg\n", net_id);
		return -1;
	}

	API_LOCK(&pnet->net_lock);
	if (update_port_cfg_mfd(pnet, net_in, net_out,
		&need_update_port, &need_update_poke) < 0) {
		printf("Update port cfg err\n");
		return -1;
	}
	if (need_update_port) {
		update_run_dags_struct_port_mfd(pnet);
	}
	if (need_update_poke) {
		update_run_dags_struct_poke_mfd(pnet);
	}
	API_UNLOCK(&pnet->net_lock);

	return 0;
}

int nnctrl_run_net_by_mfd(int net_id,
	struct net_result *net_ret, struct net_run_cfg *net_run)
{
	struct nnctrl_info *pctl = NULL;
	struct net_desc *pnet = NULL;
	float vp_time_us = 0.0f;
	int rval = 0;

	pctl = get_nnctrl_global_context();
	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	if (!pnet->net_load_done) {
		printf("Network id: %d is not load before run\n", net_id);
		return -1;
	}

	API_LOCK(&pnet->net_lock);

	if (net_run) {
		pnet->split_num = net_run->split_num_run;
		if (net_run->single_dag_run) {
			rval = single_dag_run_mfd(pctl, pnet, net_id, &vp_time_us);
		} else if (pnet->split_num > 1) {
			rval = split_net_run_mfd(pctl, pnet, net_id, pnet->split_num, &vp_time_us);
		} else {
			rval = all_dag_run_mfd(pctl, pnet, net_id, &vp_time_us);
		}
	} else {
		rval = all_dag_run_mfd(pctl, pnet, net_id, &vp_time_us);
	}

	API_UNLOCK(&pnet->net_lock);

	if (!rval && net_ret) {
		net_ret->vp_time_us = vp_time_us;
	}

	return rval;
}

