/*******************************************************************************
 * nnctrl.c
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
#include "rundags.h"
#include "rundags_mfd.h"
#include "debug.h"
#include "utils.h"

static struct nnctrl_version G_version = {
	.major = NNCTRL_LIB_MAJOR,
	.minor = NNCTRL_LIB_MINOR,
	.patch = NNCTRL_LIB_PATCH,
	.cavalry_parser = CAVALRY_GEN_VERSION,
	.mod_time = 0x20200527,
	.description = "Neural Network Control Library",
};

static struct nnctrl_info G_nnctrl = {0};

struct nnctrl_info *get_nnctrl_global_context(void)
{
	return &G_nnctrl;
}

int nnctrl_init(int fd_cav, uint8_t verbose)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct nnctrl_version *pver = &G_version;
	uint64_t audio_clk = 0;
	int rval = 0;

	if (fd_cav < 0) {
		printf("Invalid nnctrl init param\n");
		return -1;
	}

	if (pctl->init_done) {
		printf("Library is inited already, do not do it again\n");
		return -1;
	} else {
		if (!pctl->init_audio_clk_done) {
			audio_clk = get_audio_clk(fd_cav);
			pctl->audio_clk_mhz = (float)((double) audio_clk / 1000000);
			pctl->init_audio_clk_done = 1;
		}
		rval = pthread_mutex_init(&pctl->list_lock, NULL);
		if (rval < 0) {
			perror("nnctrl list lock init");
		} else {
			pctl->fd_cav = fd_cav;
			pctl->verbose = !!verbose;
			pctl->candidate_id = 0;
			pctl->init_done = 1;
			INIT_LIST_HEAD(&pctl->net_list);
		}
	}

	if ((rval == 0) && (verbose)) {
		printf("%s: %u.%u.%u, mod-time: 0x%x, built-time: %s - %s\n",
			pver->description, pver->major, pver->minor, pver->patch, pver->mod_time,
			__DATE__, __TIME__);
		printf("Cavalry Parser Version: %u.%u.%u\n",
			(pver->cavalry_parser & 0xff000000) >> 24,
			(pver->cavalry_parser & 0x00ff0000) >> 16,
			(pver->cavalry_parser & 0x0000ffff));
		printf("Audio Clock: %lu Hz\n", audio_clk);
	}

	return rval;
}

int nnctrl_get_version(struct nnctrl_version *ver)
{
	struct nnctrl_version *pver = &G_version;

	if (ver == NULL) {
		printf("Version pointer is NULL\n");
		return -1;
	}

	memcpy(ver, pver, sizeof(struct nnctrl_version));
	return 0;
}

int nnctrl_suspend_net(uint8_t quit_all)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct cavalry_early_quit early_quit = {0};
	int rval = 0;

	early_quit.early_quit_all = !!quit_all;
	if (ioctl(pctl->fd_cav, CAVALRY_EARLY_QUIT, &early_quit) < 0) {
		perror("CAVALRY_EARLY_QUIT");
		rval = -1;
	}

	return rval;
}

int nnctrl_init_net(struct net_cfg *net_cf, struct net_input_cfg *net_in,
	struct net_output_cfg *net_out)
{
	struct nnctrl_info *pctl = &G_nnctrl;
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
	if (check_io_num(net_in, net_out) < 0) {
		return -1;
	}

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
		pnet->use_memfd = 0;
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

		if (get_port_cfg(pnet, net_in, net_out) < 0) {
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

int nnctrl_load_net(int net_id, struct net_mem *net_m,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	int rval = 0;

	if (net_m == NULL) {
		printf("Invalid nnctrl load net param\n");
		return -1;
	}
	if ((net_m->virt_addr == NULL) || (net_m->mem_size == 0)) {
		printf("Invalid nnctrl net memory param %p, %u\n",
			net_m->virt_addr, net_m->mem_size);
		return -1;
	}
	if (check_io_num(net_in, net_out) < 0) {
		return -1;
	}
	if ((net_in == NULL) || (net_out == NULL)) {
		printf("Warning: Not specify net_io pointer,"
			" should get net_io by call nnctrl_get_net_io_cfg() later\n");
	}

	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	pnet->virt_addr = net_m->virt_addr;
	pnet->phy_addr = net_m->phy_addr;
	pnet->mem_size = net_m->mem_size;

	do {
		if (load_dvi_image_bin(pnet) < 0) {
			printf("Load dvi image err\n");
			rval = -1;
			break;
		}

		if (set_port_cfg(pnet, net_in, net_out) < 0) {
			printf("set port cfg failed!\n");
			rval = -1;
			break;
		}
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

	if (set_run_dags_struct(pnet) < 0) {
		printf("init run dags ioctl struct err\n");
		rval = -1;
	}

	if (rval == 0) {
		pnet->net_load_done = 1;
	}

	return rval;
}

int nnctrl_get_net_io_cfg(int net_id,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	uint32_t i = 0, is_loaded = 0;

	if ((net_in == NULL) && (net_out == NULL)) {
		printf("Both of input and output are NULL\n");
		return -1;
	}

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
				net_in->in_desc[i].addr = pnet->phy_addr + pnet->net_prt_in[i].port_dram_addr;
				net_in->in_desc[i].virt = pnet->virt_addr + pnet->net_prt_in[i].port_dram_addr;
			} else {
				net_in->in_desc[i].addr = 0;
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
				net_out->out_desc[i].addr = pnet->phy_addr + pnet->net_prt_out[i].port_dram_addr;
				net_out->out_desc[i].virt = pnet->virt_addr + pnet->net_prt_out[i].port_dram_addr;
			} else {
				net_out->out_desc[i].addr = 0;
				net_out->out_desc[i].virt = NULL;
			}
		}
	}
	API_UNLOCK(&pnet->net_lock);

	return 0;
}

int nnctrl_set_net_io_cfg(int net_id,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	uint32_t need_update_port = 0, need_update_poke = 0;

	if ((net_in == NULL) && (net_out == NULL)) {
		printf("Both of input and output are NULL\n");
		return -1;
	}

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
	if (update_port_cfg(pnet, net_in, net_out,
		&need_update_port, &need_update_poke) < 0) {
		printf("Update port cfg err\n");
		return -1;
	}
	if (need_update_port) {
		update_run_dags_struct_port(pnet);
	}
	if (need_update_poke) {
		update_run_dags_struct_poke(pnet);
	}
	API_UNLOCK(&pnet->net_lock);

	return 0;
}

int nnctrl_query_dvi(int net_id, uint32_t dvi_id, struct net_dvi_cfg *net_dvi)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	dvi_node_t *node = NULL;
	uint32_t i = 0, found = 0;
	int rval = 0;

	if (net_dvi == NULL) {
		printf("Invalid NULL pointer of dvi\n");
		return -1;
	}

	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	if (!pnet->net_load_done) {
		printf("Network id: %d is not load before run\n", net_id);
		return -1;
	}

	if (dvi_id > pnet->header.dvi_num) {
		printf("Invalid dvi id: %u, max dvi num: %u on net: %u\n",
			dvi_id, pnet->header.dvi_num, net_id);
		return -1;
	}
	if ((!pnet->virt_addr) || (!pnet->phy_addr)) {
		printf("Invalid Net mem virt: %p, phy: 0x%x, Not load_net yet? \n",
			pnet->virt_addr, pnet->phy_addr);
		return -1;
	}

	node = pnet->dvi_node_list;
	for (i = 0; i < pnet->header.dvi_num; i++, node++) {
		if (dvi_id == node->dvi_desc.dvi_id) {
			net_dvi->virt = pnet->virt_addr + node->dvi_dram_addr;
			net_dvi->addr = pnet->phy_addr + node->dvi_dram_addr;
			net_dvi->dvi_desc = node->dvi_desc;
			found = 1;
			break;
		}
	}
	if (!found) {
		rval = -1;
		printf("Not found dvi_id: %u\n", dvi_id);
	}

	return rval;
}

int nnctrl_run_net(int net_id,
	struct net_result *net_ret, struct net_run_cfg *net_run,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	float vp_time_us = 0.0f;
	uint32_t need_update_port = 0, need_update_poke = 0;
	int rval = 0;

	if (check_io_num(net_in, net_out) < 0) {
		return -1;
	}

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
	if (update_port_cfg(pnet, net_in, net_out,
		&need_update_port, &need_update_poke) < 0) {
		printf("Update input port cfg err\n");
		return -1;
	}

	if (need_update_port) {
		update_run_dags_struct_port(pnet);
	}
	if (need_update_poke) {
		update_run_dags_struct_poke(pnet);
	}
	if (update_run_dags_struct_loop_cnt(pnet, net_run) < 0) {
		printf("Update net run err\n");
		return -1;
	}
	if (net_run) {
		pnet->split_num = net_run->split_num_run;
		if (net_run->single_dag_run) {
			rval = single_dag_run(pctl, pnet, net_id, &vp_time_us);
		} else if (pnet->split_num > 1) {
			rval = split_net_run(pctl, pnet, net_id, pnet->split_num, &vp_time_us);
		} else {
			rval = all_dag_run(pctl, pnet, net_id, &vp_time_us);
		}
	} else {
		rval = all_dag_run(pctl, pnet, net_id, &vp_time_us);
	}

	API_UNLOCK(&pnet->net_lock);

	if (!rval && net_ret) {
		net_ret->vp_time_us = vp_time_us;
	}

	return rval;
}

int nnctrl_resume_net(int net_id, struct net_result *net_ret)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	struct cavalry_run_dags *org = NULL;
	struct cavalry_run_dags *run = NULL;
	struct timeval tv1, tv2;
	unsigned long tv_diff = 0;
	unsigned long diff_tick = 0, total_tick = 0;
	float vp_time_us = 0.0f;
	uint32_t nfinished = 0, npartial = 0;
	uint32_t total = 0, n = 0, last_idx = 0;
	uint32_t split_num = 0, head = 0;
	int rval = 0;

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
	run = &pnet->partial_dags;
	org = &pnet->run_dags;
	total = org->dag_cnt;
	split_num = pnet->split_num;
	nfinished = pnet->dag_idx;
	npartial = total /split_num;
	last_idx = npartial * (split_num - 1);

	if (pnet->print_time) {
		gettimeofday(&tv1, NULL);
	}
	while (nfinished < total) {
		run->rval = 0;
		run->start_tick = 0;
		run->end_tick = 0;
		run->finish_dags = 0;
		if (split_num > 1) {
			if (nfinished < npartial) {
				head = nfinished;
			} else {
				head = nfinished % split_num;
			}
			if (nfinished >= last_idx) {
				n = total - nfinished;
			} else if (head) {
				/* in the middle of partial net */
				n = npartial - head;
			} else {
				n = npartial;
			}
		} else {
			n = total - nfinished; /* run all left dags once */
		}
		run->dag_cnt = n;
		memcpy(&run->dag_desc[0], &org->dag_desc[nfinished],
			n * sizeof(struct cavalry_dag_desc));

		if (ioctl(pctl->fd_cav, CAVALRY_RUN_DAGS, run) < 0) {
			if (errno == EINTR) {
				rval = -EINTR;
			} else {
				perror("CAVALRY_RUN_DAGS");
				rval = -1;
			}
		}

		diff_tick = run->start_tick <= run->end_tick ? (run->end_tick - run->start_tick) :
			(0xFFFFFFFF - run->start_tick + run->end_tick);
		vp_time_us = (float)diff_tick / pctl->audio_clk_mhz;
		total_tick += diff_tick;

		if (rval == 0) {
			nfinished += run->finish_dags;
			if (run->finish_dags != run->dag_cnt) {
				if (run->finish_dags < run->dag_cnt) {
					pnet->dag_idx = nfinished;
					rval = -EAGAIN;
				} else {
					printf("Abnormal run dags: %u / %u\n", run->finish_dags, run->dag_cnt);
					rval = -1;
				}
			}
		}
		if (pnet->print_time) {
			printf("Net_id: %d, Dag_cnt: %u, vp_ticks: %lu, vp_time: %lu us. [Resume %s]\n",
				net_id, run->finish_dags, diff_tick, (unsigned long)vp_time_us,
				(split_num > 1) ? "Split": " ");
		}
		if (rval < 0) {
			break;
		}
	}
	if (pnet->print_time) {
		gettimeofday(&tv2, NULL);
	}
	API_UNLOCK(&pnet->net_lock);

	vp_time_us = total_tick / pctl->audio_clk_mhz;
	if (pnet->print_time) {
		tv_diff = (unsigned long) 1000000 * (unsigned long) (tv2.tv_sec - tv1.tv_sec) +
			(unsigned long) (tv2.tv_usec - tv1.tv_usec);
		printf("Net_id: %d, Dags: %u / %u, vp_ticks: %lu, vp_time: %lu us, arm_time: %lu us. [Resume]\n",
			net_id, nfinished, org->dag_cnt, total_tick, (unsigned long)vp_time_us, tv_diff);
	}

	if (!rval && (net_ret != NULL)) {
		net_ret->vp_time_us = vp_time_us;
	}

	return rval;
}

int nnctrl_dump_net(int net_id, const char *path)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;

	if (!path) {
		printf("path is NULL\n");
		return -1;
	}

	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}
	if (!pnet->net_load_done) {
		printf("Network id: %d is not load before dump\n", net_id);
		return -1;
	}

	if (dump_blob_and_dvi(pnet, path) < 0) {
		printf("dump mode enable. dump blob and dvi failed!\n");
		return -1;
	}

	return 0;
}

int nnctrl_exit_net(int net_id)
{
	struct nnctrl_info *pctl = &G_nnctrl;
	struct net_desc *pnet = NULL;
	dvi_node_t *node = NULL;
	uint32_t i = 0;
	int rval = 0;

	pnet = get_net_desc(pctl, net_id);
	if (!pnet) {
		printf("Not found net desc\n");
		return -1;
	}

	node = pnet->dvi_node_list;
	if (node != NULL) {
		for (i = 0; i < pnet->header.dvi_num; i++) {
			if (node->in_port) {
				free(node->in_port);
				node->in_port = NULL;
			}
			if (node->out_port) {
				free(node->out_port);
				node->out_port = NULL;
			}
			node ++;
		}
		free(pnet->dvi_node_list);
		pnet->dvi_node_list = NULL;
	}

	API_LOCK(&pctl->list_lock);
	list_del(&pnet->node);
	API_UNLOCK(&pctl->list_lock);

	if (pnet) {
		free(pnet);
		pnet = NULL;
	}

	return rval;
}

void nnctrl_exit(void)
{
	struct nnctrl_info *pctl = &G_nnctrl;

	pctl->fd_cav = -1;
	pctl->init_done = 0;
}
