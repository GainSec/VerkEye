/*******************************************************************************
 * parser.h
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

#ifndef _PARSER_H_
#define _PARSER_H_

#include <stdint.h>
#include "cavalry_gen.h"
#include "nnctrl_priv.h"

int check_cavalry_gen_bin_header(struct net_desc *pnet);

int check_io_num(struct net_input_cfg *net_in, struct net_output_cfg *net_out);

int gen_net_sub_parent_port(struct net_desc *pnet);
void gen_net_parent_port_addr(struct net_desc *pnet);

void set_sub_and_parent_port_mfd(struct net_desc *pnet);

int parse_dvi_desc(struct net_desc *pnet);
void update_layer_name(struct net_desc *pnet);

int allocate_port_mem(struct net_desc *pnet);

int load_dvi_image_bin(struct net_desc *pnet);

void show_net_parent_port(struct net_desc *pnet);

void split_parent_cfg_to_sub(parent_port_desc_t *prt_port);
int split_parent_rt_flip_to_sub(parent_port_desc_t *prt_port, uint32_t is_output);
void split_parent_pitch_to_sub(parent_port_desc_t *prt_port);
int split_parent_addr_to_sub(parent_port_desc_t *prt_port);


int get_port_cfg(struct net_desc *pnet,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out);
int set_port_cfg(struct net_desc *pnet,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out);
int update_port_cfg(struct net_desc *pnet,
	struct net_input_cfg *net_in, struct net_output_cfg *net_out,
	uint32_t *need_update_port, uint32_t *need_update_poke);

#endif
