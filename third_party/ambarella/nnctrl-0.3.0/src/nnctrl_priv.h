/*******************************************************************************
 * nnctrl_priv.h
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

#ifndef _NNCTRL_PRIV_H_
#define _NNCTRL_PRIV_H_

#include <stdint.h>
#include <pthread.h>
#include <cavalry_ioctl.h>
#include <cavalry_gen.h>
#include "nnctrl.h"
#include "list.h"

#define MAX_SUB_PORT_NUM	(8)

#define MAX_DEPENDENCY_NUM	(128)
#define MAX_BLOB_NUM	(1024)
#define MAX_DVI_DEBUG_NUM	(256)

#define S32_VALUE_MAX (0x7fffffff)

typedef enum port_type_s {
	PORT_TYPE_INVALID = 0,
	PORT_TYPE_INPUT = 1,
	PORT_TYPE_ITM = 2,
	PORT_TYPE_ITM_FROM_OUT = 3,
	PORT_TYPE_OUTPUT = 4,
} port_type_t;

typedef struct port_desc_s {
	io_descriptor_t io_desc;  /* Do not change this field order */

	/* extra */
	uint32_t dvi_id;
	uint32_t no_mem : 1;
	uint32_t rotate_flip_in_dvi : 5;
	uint32_t reserved_0 : 2;
	uint32_t rotate_flip_bitmap : 5;
	uint32_t reserved_1 : 3;
	uint32_t update_pitch : 16;

	struct cavalry_port_desc *run_port;

	port_type_t port_type;
	uint32_t port_dram_fd;
	uint32_t port_dram_addr;
	uint32_t port_dram_size;
	uint32_t port_remain_size;
	uint32_t needed_by_dvi_cnt;
	uint32_t needed_by_dvi[MAX_DEPENDENCY_NUM];
} port_desc_t;

typedef struct parent_port_desc_s {
	const char *name;

	uint32_t dvi_id;
	uint32_t no_mem : 1;
	uint32_t rotate_flip_bitmap : 5;
	uint32_t reserved_0 : 2;
	uint32_t reserved_1 : 8;
	uint32_t update_pitch : 16;

	struct io_dim dim;
	struct io_data_fmt data_fmt;

	int port_dram_fd;
	uint32_t port_dram_addr;
	uint32_t port_dram_size;
	uint32_t port_size;
	uint32_t port_remain_size;
	uint32_t needed_by_dvi_cnt;
	uint32_t needed_by_dvi[MAX_DEPENDENCY_NUM];

	uint32_t sub_port_num;
	port_desc_t *sub_port[MAX_SUB_PORT_NUM];
} parent_port_desc_t;

typedef struct dvi_node_s {
	dvi_desc_t dvi_desc;  /* Do not change this field order */

	/* extra */
	uint32_t dvi_file_pos;
	uint32_t dvi_dram_addr;

	port_desc_t *in_port;
	port_desc_t *out_port;
} dvi_node_t;

/*******************************************/

struct net_desc {
	FILE *net_fp;
	const void *net_feed_virt;
	uint32_t net_feed_virt_pos;

	struct list_head node;
	int net_id;
	uint32_t verbose : 1; /* net verbose */
	uint32_t reuse_mem : 1; /* if reuse mem */
	uint32_t print_time : 1; /* timer verbose */
	uint32_t net_init_done : 1; /* set this after init net, check this before load/run/exit net */
	uint32_t net_load_done : 1; /* set this after load net, check this before run/exit net */
	uint32_t use_memfd : 1;
	uint32_t reserved_0 : 26;

	uint8_t *virt_addr; /* The virtual address mmap by app for network */
	uint32_t phy_addr; /* The absolute physical address */
	uint32_t mem_size; /* The memory size assigned by app */

	uint32_t mem_offset; /* The offset base on virt_addr or phy_addr.
	                    * app access: virt_addr + mem_offset;
	                    * ioctl info:  phy_addr + mem_offset; */
	int mem_fd; /* DMABuf fd */

	uint32_t net_loop_cnt;
	uint32_t dvi_mem_align_total;
	uint32_t blob_mem_align_total;
	uint32_t dvi_mem_total;
	uint32_t blob_mem_total; /* Consume DRAM size */
	uint32_t blob_bw_total; /* Real bandwith on blob (dvi's input and output) */

	uint32_t blob_num;
	uint32_t blob_size[MAX_BLOB_NUM];
	uint32_t blob_addr[MAX_BLOB_NUM];
	const char *blob_name[MAX_BLOB_NUM];

	cavalry_gen_header_t header;
	dvi_node_t *dvi_node_list;

	uint32_t net_in_num;
	uint32_t net_out_num;
	port_desc_t *net_in[MAX_IO_NUM];
	port_desc_t *net_out[MAX_IO_NUM];

	uint32_t net_prt_in_num;
	uint32_t net_prt_out_num;
	parent_port_desc_t net_prt_in[MAX_IO_NUM];
	parent_port_desc_t net_prt_out[MAX_IO_NUM];

	pthread_mutex_t net_lock; /* lock for net_io_cfg */

	uint32_t split_num;
	uint32_t dag_idx; /* run net when interrupt, remeber finished dags then */
	/* Add additional field above. Do not add any filed after struct cavalry_run_dags */
	struct cavalry_run_dags run_dags;
	struct cavalry_run_dags partial_dags;

	struct cavalry_run_dags_mfd run_dags_mfd;
	struct cavalry_run_dags_mfd partial_dags_mfd;
};

struct nnctrl_info {
	int32_t fd_cav;
	uint32_t verbose : 1; /* nnctrl verbose */
	uint32_t init_done : 1;
	uint32_t init_audio_clk_done : 1;
	uint32_t reserved_0 : 29;

	pthread_mutex_t list_lock; /* lock for net_list */

	float audio_clk_mhz; /* default is 12.288 MHz */
	int candidate_id;
	struct list_head net_list;
};

struct nnctrl_info *get_nnctrl_global_context(void);

#endif
